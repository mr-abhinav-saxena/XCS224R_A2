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
        use_tb: bool, # whether to use tensorboard for logging
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
        self.critic_target.load_state_dict(self.critic.state_dict()) # initialize target critic parameters to critic parameters

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train() # train the actor and critic networks
        
        # Setting target critic to training mode ensures that the target critic network is in the same mode as the critic network, which is important for consistency during training. The target critic network is used to compute the target Q-values for the critic update, and it should be in training mode to ensure that any layers that behave differently during training (e.g., dropout, batch normalization) are handled correctly.
        self.critic_target.train() # set the target critic network to training mode (even though it won't be updated by gradients)

    def train(self, training: bool = True) -> None: # set the actor and critic networks to training or evaluation mode
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs: np.ndarray, eval_mode: bool) -> np.ndarray:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        dist = self.actor(obs.unsqueeze(0)) # get the action distribution from the actor network
        if eval_mode:
            action = dist.mean # get the mean action from the distribution (for evaluation)
        else:
            action = dist.sample(clip=None) # sample an action from the distribution (without clipping)
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
        # Sample next state actions from the policy
        with torch.no_grad(): # no gradient computation for the target critic update
            next_action_dist = self.actor(next_obs) # get the action distribution for the next states from the actor network
            next_action = next_action_dist.sample(clip=None) # sample actions for the next states from the action distribution without clipping (we will handle clipping later when we compute the target Q-values)

            # Compute target Q-values using the target critic network
            target_q_values = self.critic_target(next_obs, next_action) # get the target Q-value(s) [for each observation, action in the batch] from the target critic network(s) for the next states and sampled actions -> list of [batch, 1] tensors, one for each critic
            

            # Randomly sample two target critics for the target Q-value computation to mitigate overestimation bias
            if len(target_q_values) > 1: # if there are multiple critics
                idx1, idx2 = random.sample(range(len(target_q_values)), 2)
                min_target_q_value = torch.min(target_q_values[idx1], target_q_values[idx2]) # take the minimum Q-value across the two randomly sampled critics -> [batch, 1] tensor
            else: # if there is only one critic
                # n_critic = 1 case is not tested yet (especially check the shape of target_q_value). It should be [batch, 1]. If not, fix the shape.
                min_target_q_value = target_q_values[0] # [batch, 1] tensor

            # Compute Bellman targets for critic update: y = r + γ * min(Q_target) -> [batch,]
            bellman_targets = reward + discount * min_target_q_value # compute the Bellman targets using the rewards and discounted minimum target Q-values -> [batch, 1]

        # Compute the critic loss: L = Σ_i (Q_i(o_t, a_t) - sg(y))^2 -> [batch, num_critics]
        current_q_values = self.critic(obs, action) # get the current Q-value(s) -> list of [batch, 1] tensors, one for each critic
        current_q_values = torch.cat(current_q_values, dim=-1) # concatenate the Q-values from multiple critics along the last dimension -> [batch, num_critics]
        
        # Expand the Bellman targets to match the shape of current Q-values for loss computation
        bellman_targets = bellman_targets.expand_as(current_q_values) # expand targets to match [batch, num_critics]
        
        critic_loss = F.mse_loss(current_q_values, bellman_targets, reduction='none') # compute the MSE loss for each critic -> [batch, num_critics]
        critic_loss = critic_loss.mean() # average the loss over the batch and critics -> scalar

        # Take a gradient step with respect to the critic parameters
        self.critic_opt.zero_grad() # zero the gradients of the critic optimizer
        critic_loss.backward() # backpropagate the critic loss
        self.critic_opt.step() # update the critic parameters with the optimizer step

        # Update the target critic parameters using exponential moving average using utils.soft_update_params function
        # Q_target = (1 - τ) * Q_target + τ * Q
        with torch.no_grad(): # no gradient computation for the target critic update
            for critic, target_critic in zip(self.critic.critics, self.critic_target.critics):
                utils.soft_update_params(critic, target_critic, self.critic_target_tau) # update the target critic parameters using soft update

        # Log the critic loss and mean Bellman targets for monitoring
        metrics["critic_loss"] = critic_loss.item() # log the critic loss as a scalar
        metrics["mean_bellman_target"] = bellman_targets.mean().item() # log the mean Bellman target across the batch as a scalar
        
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
        # Sample an action from the actor policy a′_t ∼ π_θ(o_t) using rsample() for reparameterization trick
        # rsample() allows gradients to flow through the sampling operation,
        # which is essential for policy gradient methods
        # sample() would break the gradient flow and prevent proper policy updates
        action_dist = self.actor(obs)
        action = action_dist.rsample()

        # Compute the objective that optimizes the actor to maximize the Q-value estimates from the critics
        # loss = − 1/N * (Σ_i (Q_i(o_t,a′_t))
        q_values = self.critic(obs, action) # get the Q-value(s) from the critic(s) for the sampled actions -> list of [batch, 1] tensors, one for each critic
        q_values = torch.cat(q_values, dim=-1) # concatenate the Q-values from multiple critics along the last dimension -> [batch, num_critics]
        actor_loss = -q_values.mean() # compute the negative mean Q-value across critics and batch -> scalar

        # Take a gradient step with respect to the actor parameters
        self.actor_opt.zero_grad() # zero the gradients of the actor optimizer
        actor_loss.backward() # backpropagate the actor loss
        self.actor_opt.step() # update the actor parameters with the optimizer step

        # Log the actor loss for monitoring
        metrics["actor_loss"] = actor_loss.item() # log the actor loss as a scalar

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
        # Behavior Cloning: Learn to imitate expert demonstrations
        # Objective: maximize log probability of expert actions
        # Loss: L = -log π(a_t | o_t)
        # We minimize the negative log-likelihood, which is equivalent to maximizing the log probability
     
        action_dist = self.actor(obs) # get the action distribution from the actor network
        
        # Compute the negative log-likelihood of the expert actions
            # For multivariate distributions, log_prob returns [batch, action_dim]
            # We need to sum over action dimensions to get joint log probability,
            # then average over the batch
            # .sum(-1, keepdim=True) sums over action_dim -> [batch, 1]
            # .mean() averages over batch -> scalar
        log_prob = action_dist.log_prob(action).sum(dim=-1)
        loss = -log_prob.mean()

        # Update the actor using the behavior cloning loss
        self.actor_opt.zero_grad() # zero the gradients of the actor optimizer
        loss.backward() # backpropagate the loss
        self.actor_opt.step() # update the actor parameters with the optimizer step

         # Log the behavior cloning loss for monitoring
        metrics["bc_loss"] = loss.item()

        # *** END CODE HERE ***

        return metrics
