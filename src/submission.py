import hydra
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Union, Optional, Any

import utils

class Actor(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...], hidden_dim: int, std: float = 0.1) -> None:
        super().__init__()

        self.std = std
        self.policy = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )

        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor) -> utils.TruncatedNormal:
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * self.std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...], num_critics: int, hidden_dim: int) -> None:
        super().__init__()

        self.critics = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(obs_shape[0] + action_shape[0], hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_critics)
            ]
        )

        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:
        h_action = torch.cat([obs, action], dim=-1)
        return [critic(h_action) for critic in self.critics]


class ACAgent:
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: torch.device,
        lr: float,
        hidden_dim: int,
        num_critics: int,
        critic_target_tau: float,
        stddev_clip: float,
        use_tb: bool,
    ) -> None:
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.stddev_clip = stddev_clip

        # models
        self.actor = Actor(obs_shape, action_shape, hidden_dim).to(device)

        self.critic = Critic(obs_shape, action_shape, num_critics, hidden_dim).to(
            device
        )
        self.critic_target = Critic(
            obs_shape, action_shape, num_critics, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs: np.ndarray, eval_mode: bool) -> np.ndarray:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        dist = self.actor(obs.unsqueeze(0))
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.cpu().numpy()[0]

    def update_critic(self, batch: Tuple[Any, ...]) -> Dict[str, float]:
        """
        This function updates the critic and target critic parameters.

        Args:

        batch:
            A batch of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the critic
                 loss, or the mean Bellman targets.
        """

        metrics = dict()

        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # *** START CODE HERE ***
        # *** END CODE HERE ***

        #####################
        return metrics

    def update_actor(self, batch: Tuple[Any, ...]) -> Dict[str, float]:
        """
        This function updates the policy parameters.

        Args:

        batch:
            A batch of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the actor
                 loss.
        """
        metrics = dict()

        obs, _, _, _, _ = utils.to_torch(batch, self.device)

        # *** START CODE HERE ***
        # *** END CODE HERE ***

        return metrics

    def bc(self, batch: Tuple[Any, ...]) -> Dict[str, float]:
        """
        This function updates the policy with end-to-end
        behavior cloning

        Args:

        batch:
            A batch of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the loss.
        """

        metrics = dict()

        obs, action, _, _, _ = utils.to_torch(batch, self.device)

        # *** START CODE HERE ***
        # *** END CODE HERE ***

        return metrics
