import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig

import numpy as np
import torch
from dm_env import specs
from typing import Any, Iterator, Optional, Dict
from omegaconf import DictConfig

import mw
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec: Any, action_spec: Any, cfg: DictConfig) -> Any:
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg: DictConfig) -> None:
        self.work_dir = Path(HydraConfig.get().runtime.output_dir)
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self) -> None:
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = mw.make(self.cfg, self.work_dir, False)
        self.eval_env = mw.make(self.cfg, self.work_dir, True)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.demo_storage = ReplayBufferStorage(data_specs,
                                                self.work_dir / 'demos')

        self.demo_loader = make_replay_loader(
            self.work_dir / 'demos', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)

        self._demo_iter = None

        from shutil import copytree

        copytree(str(Path.cwd() / "demos/"), str(self.work_dir / "demos"), dirs_exist_ok=True)
        copytree(str(Path.cwd() / "demos/"), str(self.work_dir / 'buffer'), dirs_exist_ok=True)

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def global_episode(self) -> int:
        return self._global_episode

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self) -> Iterator[Any]:
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def demo_iter(self) -> Iterator[Any]:
        if self._demo_iter is None:
            self._demo_iter = iter(self.demo_loader)
        return self._demo_iter

    def eval(self, num_eval_episodes: int = 100) -> None:
        step, episode, total_reward, total_success = 0, 0, 0, 0
        eval_until_episode = utils.Until(num_eval_episodes or self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()

            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)

                total_reward += time_step.reward
                step += 1

            total_success += time_step.reward > 0.0
            episode += 1

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_success', total_success / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('eval_total_time', self.timer.total_time())

    def train(self) -> None:
        # predicates

        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        # pretrain with BC
        for pretrain_step in range(2000):
            metrics = self.agent.bc(next(self.demo_iter))
            self.logger.log_metrics(metrics, pretrain_step, ty='pretrain')

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # log stats
                elapsed_time, total_time = self.timer.reset()
                episode_frame = episode_step * self.cfg.action_repeat
                with self.logger.log_and_dump_ctx(self.global_frame,
                                                    ty='train') as log:
                    log('fps', episode_frame / elapsed_time)
                    log('total_time', total_time)
                    log('episode_reward', episode_reward)
                    log('episode_length', episode_frame)
                    log('episode', self.global_episode)
                    log('buffer_size', len(self.replay_storage))
                    log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        eval_mode=self.global_frame <= self.cfg.warmup)

            # try to update the agent
            for _ in range(self.cfg.utd):
                critic_metrics = self.agent.update_critic(next(self.replay_iter))

            self.logger.log_metrics(critic_metrics, self.global_frame, ty='critic')

            if self.global_frame > self.cfg.warmup:
                actor_metrics = self.agent.update_actor(next(self.replay_iter))

                if self.global_step % self.cfg.bc_freq == 0:
                    bc_metrics = self.agent.bc(next(self.demo_iter))

                self.logger.log_metrics(actor_metrics, self.global_frame, ty='actor')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self) -> None:
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self) -> None:
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg: DictConfig) -> None:
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
