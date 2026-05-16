#!/usr/bin/env python3
import inspect
from pathlib import Path
import unittest
import argparse
from graderUtil import graded, CourseTestRunner, GradedTestCase
from subprocess import Popen
from subprocess import DEVNULL, STDOUT, check_call
import torch
import torch.nn as nn

import numpy as np
import pickle
import os
import random

import gymnasium as gym

from dm_env import specs

from replay_buffer import ReplayBufferStorage, make_replay_loader
import mw

import hydra
from hydra import compose, initialize

import tensorflow.compat.v1 as tf

#########
# UTILS #
#########

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


def load_config_static(overrides):

    with initialize(version_base=None, config_path="cfgs"):
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
        return cfg


def parse_file(file):
    eval_returns = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == "eval/episode_success":
                eval_returns.append(v.simple_value)

    max_average_return = np.max(np.array(eval_returns))
    return max_average_return


def get_scores(rootdir, subquestion):

    score = 0

    for root, dirs, filelist in os.walk(rootdir):
        # Skip hidden or system folders like __MACOSX
        if "__MACOSX" in root or "/." in root:
            continue

        if (root.endswith("_agent.num_critics=2,utd=1/tb") and subquestion == "i") or \
            (root.endswith("_agent.num_critics=10,utd=5/tb") and subquestion == "ii"):

            for file in filelist:
                if file.startswith(".") or "MACOSX" in file:
                    continue

            for file in filelist:
                if "event" in file:
                    try:
                        score = parse_file(root + "/" + file)
                    except:
                        print(f"Error parsing file {file}")
                        score = 0

    return score


#########
# TESTS
#########


class Test_2a(GradedTestCase):

    def setUp(self):

        self.student_cfg = load_config_static(
            [
                "agent.num_critics=2",
                "utd=1",
                "hydra.run.dir=./logdir_grader/run_${now:%H%M%S}_${hydra.job.override_dirname}",
                "replay_buffer_num_workers=0"
            ]
        )

        self.work_dir = Path(f"./{self.student_cfg.hydra.run.dir}")

        # create env
        self.grader_env = mw.make(self.student_cfg, self.work_dir, False)

        torch.random.manual_seed(self.student_cfg.seed)
        np.random.seed(self.student_cfg.seed)
        random.seed(self.student_cfg.seed)

        self.student_agent = make_agent(
            self.grader_env.observation_spec(),
            self.grader_env.action_spec(),
            self.student_cfg.agent,
        )

        self.demo_loader = make_replay_loader(
            self.work_dir / "demos",
            self.student_cfg.replay_buffer_size,
            self.student_cfg.batch_size,
            self.student_cfg.replay_buffer_num_workers,
            self.student_cfg.save_snapshot,
            self.student_cfg.nstep,
            self.student_cfg.discount,
        )

        self.demo_iter = iter(self.demo_loader)

        self.solution_cfg = load_config_static(
            [
                "agent.num_critics=2",
                "utd=1",
                "hydra.run.dir=./logdir_grader/run_${now:%H%M%S}_${hydra.job.override_dirname}",
                "agent._target_=solution.ACAgent",
                "replay_buffer_num_workers=0"
            ]
        )

        torch.random.manual_seed(self.solution_cfg.seed)
        np.random.seed(self.solution_cfg.seed)
        random.seed(self.solution_cfg.seed)
        try:
            self.solution_agent = make_agent(
                self.grader_env.observation_spec(),
                self.grader_env.action_spec(),
                self.solution_cfg.agent,
            )
        except Exception:
            self.solution_agent = None

        from shutil import copytree

        copytree(
            str(Path.cwd() / "demos/"), str(self.work_dir / "demos"), dirs_exist_ok=True
        )
        copytree(
            str(Path.cwd() / "demos/"),
            str(self.work_dir / "buffer"),
            dirs_exist_ok=True,
        )

    @graded(is_hidden=False)
    def test_0(self):
        """2a-0-basic: test ACAgent.bc function"""

        initial_agent_parameters = {
            k: torch.clone(v) for k, v in self.student_agent.actor.named_parameters()
        }

        self.student_agent.bc(next(self.demo_iter))

        updated_agent_parameters = {
            k: torch.clone(v) for k, v in self.student_agent.actor.named_parameters()
        }

        update_magnitude = 0

        for k, _ in initial_agent_parameters.items():
            update_magnitude += torch.abs(
                torch.sum(
                    updated_agent_parameters.get(k).data
                    - initial_agent_parameters.get(k).data
                )
            )
        self.assertGreater(update_magnitude, 0, msg="Possible issue with the ACAgent.bc function--policy parameter update magnitude was not as big as expected.")
        self.assertLess(update_magnitude, 1, msg="Possible issue with the ACAgent.bc function--policy parameter update magnitude was bigger than expected.")

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_2b(GradedTestCase):

    def setUp(self):

        self.student_cfg = load_config_static(
            [
                "agent.num_critics=2",
                "utd=1",
                "hydra.run.dir=./logdir_grader/run_${now:%H%M%S}_${hydra.job.override_dirname}",
                "replay_buffer_num_workers=0",
            ]
        )

        self.work_dir = Path(f"./{self.student_cfg.hydra.run.dir}")

        # create env
        self.grader_env = mw.make(self.student_cfg, self.work_dir, False)

        torch.random.manual_seed(self.student_cfg.seed)
        np.random.seed(self.student_cfg.seed)
        random.seed(self.student_cfg.seed)

        self.student_agent = make_agent(
            self.grader_env.observation_spec(),
            self.grader_env.action_spec(),
            self.student_cfg.agent,
        )

        self.demo_loader = make_replay_loader(
            self.work_dir / "demos",
            self.student_cfg.replay_buffer_size,
            self.student_cfg.batch_size,
            self.student_cfg.replay_buffer_num_workers,
            self.student_cfg.save_snapshot,
            self.student_cfg.nstep,
            self.student_cfg.discount,
        )

        self.demo_iter = iter(self.demo_loader)

        self.solution_cfg = load_config_static(
            [
                "agent.num_critics=2",
                "utd=1",
                "hydra.run.dir=./logdir_grader/run_${now:%H%M%S}_${hydra.job.override_dirname}",
                "agent._target_=solution.ACAgent",
                "replay_buffer_num_workers=0",
            ]
        )

        torch.random.manual_seed(self.solution_cfg.seed)
        np.random.seed(self.solution_cfg.seed)
        random.seed(self.solution_cfg.seed)
        try:
            self.solution_agent = make_agent(
                self.grader_env.observation_spec(),
                self.grader_env.action_spec(),
                self.solution_cfg.agent,
            )
        except Exception:
            self.solution_agent = None

        from shutil import copytree

        copytree(
            str(Path.cwd() / "demos/"), str(self.work_dir / "demos"), dirs_exist_ok=True
        )
        copytree(
            str(Path.cwd() / "demos/"),
            str(self.work_dir / "buffer"),
            dirs_exist_ok=True,
        )

    @graded(is_hidden=False)
    def test_0(self):
        """2b-0-basic: test ACAgent.update_critic function"""

        initial_agent_parameters = {
            k: torch.clone(v)
            for k, v in self.student_agent.critic_target.named_parameters()
        }

        self.student_agent.update_critic(next(self.demo_iter))

        updated_agent_parameters = {
            k: torch.clone(v)
            for k, v in self.student_agent.critic_target.named_parameters()
        }

        update_magnitude = 0

        for k, _ in initial_agent_parameters.items():
            update_magnitude += torch.abs(
                torch.sum(
                    updated_agent_parameters.get(k).data
                    - initial_agent_parameters.get(k).data
                )
            )
        self.assertGreater(update_magnitude, 0, msg="Possible issue with the ACAgent.update_critic function--critic_target parameter update magnitude was not as big as expected.")
        self.assertLess(update_magnitude, 1, msg="Possible issue with the ACAgent.update_critic function--critic_target parameter update magnitude was bigger than expected.")

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded(is_hidden=False)
    def test_2(self):
        """2b-2-basic: test ACAgent.update_actor function"""

        initial_agent_parameters = {
            k: torch.clone(v) for k, v in self.student_agent.actor.named_parameters()
        }

        self.student_agent.update_actor(next(self.demo_iter))

        updated_agent_parameters = {
            k: torch.clone(v) for k, v in self.student_agent.actor.named_parameters()
        }

        update_magnitude = 0

        for k, _ in initial_agent_parameters.items():
            update_magnitude += torch.abs(
                torch.sum(
                    updated_agent_parameters.get(k).data
                    - initial_agent_parameters.get(k).data
                )
            )
        self.assertGreater(update_magnitude, 0, msg="Possible issue with the ACAgent.update_actor function--actor parameter update magnitude was not as big as expected.")
        self.assertLess(update_magnitude, 5, msg="Possible issue with the ACAgent.update_actor function--actor parameter update magnitude was bigger than expected.")

    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_2ci(GradedTestCase):

    ### BEGIN_HIDE ###
    ### END_HIDE ###
    pass

class Test_2ciii(GradedTestCase):

    ### BEGIN_HIDE ###
    ### END_HIDE ###
    pass


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)

if __name__ == "__main__":
    # Parse for a specific test or mode
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test_case",
        nargs="?",
        default="all",
        help="Use 'all' (default), a specific test id like '1-3-basic', 'public', or 'hidden'",
    )
    test_id = parser.parse_args().test_case

    def _flatten(suite):
        """Recursively flatten unittest suites into individual tests."""
        for x in suite:
            if isinstance(x, unittest.TestSuite):
                yield from _flatten(x)
            else:
                yield x

    assignment = unittest.TestSuite()

    if test_id not in {"all", "public", "hidden"}:
        # Run a single specific test
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        # Discover all tests
        discovered = unittest.defaultTestLoader.discover(".", pattern="grader.py")

        if test_id == "all":
            assignment.addTests(discovered)
        else:
            # Filter tests by visibility marker in docstring ("basic" for public tests, "hidden" for hidden tests)
            keyword = "basic" if test_id == "public" else "hidden"
            filtered = [
                t for t in _flatten(discovered)
                if keyword in (getattr(t, "_testMethodDoc", "") or "")
            ]
            assignment.addTests(filtered)

    CourseTestRunner().run(assignment)
