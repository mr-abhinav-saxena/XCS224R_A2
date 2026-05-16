from collections import deque, OrderedDict
from typing import Any, NamedTuple, Tuple, Optional, Union
import pathlib

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import metaworld
import dm_env
import mujoco
import numpy as np
from dm_env import StepType, specs


class MetaWorldEnv:

    def __init__(self, name: str = "hammer-v3", action_repeat: int = 2, duration: int = 50, save_video: bool = False, workdir: Optional[pathlib.Path] = None, num_eval_episodes: int = 50) -> None:

        from metaworld.env_dict import ALL_V3_ENVIRONMENTS

        self.action_repeat = action_repeat
        self.duration = duration
        self._step = None
        self.save_video = save_video

        env_cls = ALL_V3_ENVIRONMENTS[name]

        if save_video == False:
            self._env = env_cls(reward_function_version="v2")

        else:
            self._env = env_cls(reward_function_version="v2", render_mode="rgb_array")

        self._env.max_path_length = np.inf
        self._env._freeze_rand_vec = False
        self._env._partially_observable = False
        self._env._set_task_called = True

        self.hand_init_pose = self._env.unwrapped.hand_init_pos.copy()
        self.hand_init_pose = np.array([0.1, 0.5, 0.30])

        camera_config = {
            "elevation": -22.5,
            "azimuth": 15,
            "distance": 0.75,
            "lookat": np.array([-0.15, 0.60, 0.25]),
        }
        self._env.unwrapped.mujoco_renderer.default_cam_config = camera_config

        if save_video is True:
            self._env = RecordVideo(
                self._env,
                video_folder=f"{workdir}/eval_video",
                episode_trigger=lambda x: (x + 1) % num_eval_episodes == 0,
                name_prefix="metaworld-hammer-v3",
                disable_logger=False,
            )

    def __getattr__(self, attr: str) -> Any:
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._env, attr)

    @property
    def observation_space(self) -> gym.Space:
        return self._env.observation_space

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0.0
        for _ in range(self.action_repeat):
            state, rew, terminated, truncated, info = self._env.step(action)
            state = state.astype(self._env.observation_space.dtype)
            reward += rew
            if terminated or truncated:
                break
        reward = 1.0 * info["success"]
        self._step += 1
        if self._step >= self.duration:
            truncated = True
        return state, reward, terminated, truncated, info

    def reset(self) -> np.ndarray:

        self._env.unwrapped.hand_init_pos = self.hand_init_pose + 0.03 * np.random.normal(size=3)

        _, _ = self._env.reset()
        for i in range(10):
            state, _, _, _, _ = self._env.step(np.zeros(self.action_space.shape))
            state = state.astype(self._env.observation_space.dtype)
        self._step = 0
        return state


class GymWrapper:

    def __init__(self, env: Any, act_key: str = 'action') -> None:
        self._env = env
        self._act_key = act_key

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def observation_spec(self) -> dm_env.specs.Array:
        return dm_env.specs.Array(
              shape = self._env.observation_space.shape,
              dtype = self._env.observation_space.dtype,
              name = 'observation')

    def action_spec(self) -> dm_env.specs.BoundedArray:
        return dm_env.specs.BoundedArray(
          shape = self._env.action_space.shape,
          minimum = self._env.action_space.low,
          maximum = self._env.action_space.high,
          dtype = self._env.action_space.dtype,
          name = 'action')

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return dm_env._environment.TimeStep(
            step_type=StepType.LAST if terminated or truncated else StepType.MID,
            reward=reward,
            discount=1.0,
            observation=obs,
        )

    def reset(self) -> dm_env.TimeStep:
        obs = self._env.reset()
        return dm_env._environment.TimeStep(
            step_type = StepType.FIRST,
            reward = 0.0,
            discount = 1.0,
            observation = obs
        )  


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self) -> bool:
        return self.step_type == StepType.FIRST

    def mid(self) -> bool:
        return self.step_type == StepType.MID

    def last(self) -> bool:
        return self.step_type == StepType.LAST

    def __getitem__(self, attr: Union[str, int]) -> Any:
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env: Any, dtype: Any) -> None:
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self) -> dm_env.specs.Array:
        return self._env.observation_spec()

    def action_spec(self) -> dm_env.specs.BoundedArray:
        return self._action_spec

    def reset(self) -> dm_env.TimeStep:
        return self._env.reset()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env: Any) -> None:
        self._env = env

    def reset(self) -> ExtendedTimeStep:
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action: np.ndarray) -> ExtendedTimeStep:
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step: dm_env.TimeStep, action: Optional[np.ndarray] = None) -> ExtendedTimeStep:
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self) -> dm_env.specs.Array:
        return self._env.observation_spec()

    def action_spec(self) -> dm_env.specs.BoundedArray:
        return self._env.action_spec()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def make(cfg: Any, workdir: pathlib.Path, eval: bool) -> ExtendedTimeStepWrapper:

    env = MetaWorldEnv(
        action_repeat=cfg.action_repeat,
        duration=cfg.duration,
        workdir=workdir,
        num_eval_episodes=1000,
        save_video=(True if eval == True and cfg.save_video == True else False),
    )
    env = GymWrapper(env)
    env = ActionDTypeWrapper(env, np.float32)
    env = ExtendedTimeStepWrapper(env)

    return env
