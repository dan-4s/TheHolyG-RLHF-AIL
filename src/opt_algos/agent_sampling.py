"""
This file implements trajectory sampling for the agent. Code is taken from
https://github.com/hcnoh/gail-pytorch/blob/main/models/gail.py,
and is improved.

TODO: Add feature to save the gifs of trajectories over training runs.
"""

import gym
import numpy as np
import os
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation


@torch.no_grad() # disable gradient tracking in this function.
def get_agent_trajectories(
        env: gym.Env,
        agent_model: torch.nn.Module,
        value_function: torch.nn.Module,
        discriminator: torch.nn.Module,
        num_sa_pairs: int,
        horizon: int,
        gamma: float,
        lambd: float,
        normalize_advantage: bool,
        device: torch.DeviceObjType,
    ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function queries the agent that is being trained to generate
    trajectories of state-action pairs.

    Parameters:
        env (`gym.Env`): The environment we're using to sample trajectories.
        agent_model (`torch.nn.Module`): The agent model.
        value_function (`torch.nn.Module`): The value function model (acts as a
            critic function in actor-critic).
        discriminator (`torch.nn.Module`): The discriminator model (acts as a
            Q-function).
        num_sa_pairs (`int`): The number of state-action pairs to sample.
        horizon (`int`): The max number of steps per trajectory.
        gamma (`float`): Discount factor for returns and advantage.
        lambd (`float`): Discount factor for advantage.
        normalize_advantage (`bool`): Whether to normalise the advantage
            estimate.
        device (`torch.device`): The device to put all data on.

    Returns:
        `tuple[float, torch.Tensor, ...]`: tuple with agent reward mean and the
            following tensors: observations, actions, returns, advantages, and
            gammas.
    """
    # Sample actions from the agent to build a set of state-action pairs which
    # we will then use to train the agent.
    # NOTE: we are not computing gradients on the selected actions! We are just
    #   collecting them for later! So we can disable gradients in this
    #   function.
    agent_model.eval()
    value_function.eval()
    discriminator.eval()

    # Variables to track trajectory generation.
    agent_obs = []
    agent_actions = []
    agent_returns = [] # Requires discriminator
    agent_advantages = [] # Requires the value function and discriminator
    agent_gammas = []
    agent_reward = []
    while len(agent_obs) < num_sa_pairs:
        # Run trajectories until we have a sufficient amount of data.
        obs, _ = env.reset()
        episode_done = False

        # Episode-based data structures
        episode_obs = []
        episode_acts = []
        episode_gammas = []  # Discount factor on returns and advantage
        episode_lambdas = [] # Discount factor on advantage
        
        episode_reward = 0
        num_steps = 0
        while not episode_done:
            # Run an episode to generate a trajectory.
            obs = torch.tensor(obs, device=device).float()
            agent_distr = agent_model(obs)
            agent_action = agent_distr.sample().detach().cpu().numpy()

            # Store to full dataset
            agent_obs.append(obs)
            agent_actions.append(torch.tensor(agent_action))

            # Store to episode dataset
            episode_obs.append(obs)
            episode_acts.append(torch.tensor(agent_action))
            episode_gammas.append(gamma ** num_steps)
            episode_lambdas.append(lambd ** num_steps)

            obs, reward, episode_done, _, _ = env.step(agent_action)
            episode_reward += reward
            
            if(horizon is not None and 
                num_steps >= horizon or
                len(agent_obs) >= num_sa_pairs):
                # We have hit the max episode length or generated all the data
                # we needed to generate, finish the episode.
                episode_done = True
            
            # Loop control
            num_steps += 1
        # Episode done, compute all statistics including advantage and return.
        agent_reward.append(episode_reward)

        #   First, convert all episode observations to tensors
        episode_obs     = torch.stack(episode_obs, dim=0)
        episode_acts    = torch.stack(episode_acts).to(device).float()
        episode_gammas  = torch.tensor(episode_gammas, device=device).float()
        episode_lambdas = torch.tensor(episode_lambdas, device=device).float()
        agent_gammas.append(episode_gammas)

        # Next, compute the return. The return is simply the discounted
        #   Q-function estimate of the visited state. Since the discriminator
        #   acts as a Q-function for us, we just take the discriminator output
        #   and sum over the trajectory length, from each state-action pair:
        #       sum_t(gamma^t * discriminator(s_t, a_t)).
        # NOTE: episode_costs are negative since we are maximising the expected
        #       return. TODO: Veryify that this intuition is correct!!!
        episode_costs = (-1) * torch.log(
            discriminator(episode_obs, episode_acts),
        ).squeeze().detach()
        episode_disc_costs = episode_gammas * episode_costs
        # Do a reverse cumsum to get the return for each state-action pair.
        episode_returns = episode_disc_costs - \
            episode_disc_costs.cumsum(dim=0) + episode_disc_costs.sum(dim=0)
        # Scale the returns (gets rid of extra gamma terms)
        episode_returns = episode_returns / episode_gammas
        agent_returns.append(episode_returns)

        # Next, compute the advantage. The advantage in this instance is going
        #   to be the Q-function subtracted from the value function. In this
        #   case, it is basically: discriminator(s,a) - value_function(s). In
        #   reality, we do:
        #     discriminator(s,a) - value_function(s) + gamma*value_function(s')
        #   Again, I don't know why this equation was used, it is inherited
        #   from the author of the GAIL repo this was based off of.
        #   Similarly to the returns, we then take the reverse cumsum.
        value_s = value_function(episode_obs).squeeze().detach()
        next_value = torch.cat(
            (value_s[1:], torch.tensor([0.0], device=device)),
            dim=0,
        )
        ind_advantages = episode_costs - value_s + gamma*next_value
        # Discount the individual advantages
        ind_advantages = episode_gammas * episode_lambdas * ind_advantages
        episode_advantages = ind_advantages - \
            ind_advantages.cumsum(dim=0) + ind_advantages.sum(dim=0)
        # Scale the individual advantages to remove extra discounts.
        episode_advantages = episode_advantages / (episode_gammas * episode_lambdas)
        agent_advantages.append(episode_advantages)
    
    # Convert the lists of tensors to tensors
    agent_obs = torch.stack(agent_obs, dim=0)
    agent_actions = torch.stack(agent_actions).to(device)
    agent_returns = torch.cat(agent_returns, dim=0)
    agent_advantages = torch.cat(agent_advantages, dim=0)
    agent_gammas = torch.cat(agent_gammas, dim=0)
    
    # Get relevant logging information
    agent_rwd_mean = np.mean(agent_reward).item()

    # Convert data to tensors.
    if(normalize_advantage):
        agent_advantages = (agent_advantages - agent_advantages.mean()
                            ) / agent_advantages.std()
    
    return (
        agent_rwd_mean,
        agent_obs,
        agent_actions,
        agent_returns,
        agent_advantages,
        agent_gammas,
    )

