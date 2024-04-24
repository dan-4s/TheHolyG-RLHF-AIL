"""
This file implements trajectory sampling for the agent. Code is taken from
https://github.com/hcnoh/gail-pytorch/blob/main/models/gail.py,
and is improved.

TODO: Add feature to save the gifs of trajectories at the end of training runs.
TODO: Add better docstring to the functions.
"""

import gym
import math
import numpy as np
import os
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation


@torch.no_grad() # disable gradient tracking in this function.
def stable_reverse_discounting(values: torch.Tensor, discounts: torch.Tensor):
    """
    Apply the discount vector to the values vector to get back the cummulative
    sum of discounts starting from the initial discount to the n-jth discount,
    where j is the current index of the vector. The implementation uses block
    multiplication to improve efficiency. A simple proof shows that this method
    surpasses most others in time and space complexity.

    NOTE: The torch matmul operation (@ in this code) actually has more
    numerical instability than a simple vector-vector multiple then sum! Since
    we don't actually care too much about the exact value of the advantage
    functions, we can more or less ignore these small errors in favour of a
    more stable algorithm which won't crash the training process!
    """
    size = discounts.shape[0]
    if(size == 1):
        # Corner case: a single state-action pair was observed. Handle by
        # simple return.
        return values * discounts
    m = min(50, size) # In case we just need to do a single run!
    discounts_matrix = torch.zeros(
        size=(size, m),
        device=values.device,
    )
    for idx in range(m):
        discounts_matrix[idx:, idx] = discounts[0:size-idx]
    
    output = torch.zeros_like(values, device=values.device)

    # Build the output block-by-block
    num_blocks = math.ceil(size / m)
    for idx in range(num_blocks):
        start = idx * m
        end = start + m
        if(end > size):
            end = size
        width = end - start
        try:
            output[start : end] = values[start:] @ discounts_matrix[0:size-start, 0:width]
        except:
            breakpoint()
    return output


@torch.no_grad() # disable gradient tracking in this function.
def get_episode_returns(
    discriminator: torch.nn.Module,
    episode_obs: torch.Tensor,
    episode_acts: torch.Tensor,
    episode_gammas: torch.Tensor,
):
    """
    Helper function to compute the agent returns given some
    discriminator network, along with the episode observations, actions,
    gammas, and lambdas.
    """
    # Compute the return. The return is simply the discounted Q-function
    # estimate of the visited state. Since the discriminator acts as a
    # Q-function for us, we just take the discriminator output and sum over the
    # trajectory length, from each state-action pair:
    #       sum_t(gamma^t * discriminator(s_t, a_t)).
    episode_costs = discriminator(episode_obs, episode_acts).squeeze().detach()
    
    episode_returns = stable_reverse_discounting(
        episode_costs, episode_gammas,
    )
    return episode_returns


@torch.no_grad() # disable gradient tracking in this function.
def get_episode_advantages(
    discriminator: torch.nn.Module,
    value_function: torch.nn.Module,
    episode_obs: torch.Tensor,
    episode_acts: torch.Tensor,
    episode_gammas: torch.Tensor,
    episode_lambdas: torch.Tensor,
    gamma: float,
    device: torch.DeviceObjType,
):
    """
    Helper function to compute the agent advantages given some
    value function and discriminator networks, along with the episode
    observations, actions, gammas, and lambdas.
    """
    # Compute the advantage. The advantage in this instance is going to be the
    # Q-function subtracted from the value function. In this case, it is
    # basically:
    #   discriminator(s,a) - value_function(s) + gamma*value_function(s').
    #
    # We can see this as the basic Bellman equation for a Q-function estimate.
    # This form of the advantage was inherited from the pytorch-gail repo.
    episode_costs = discriminator(episode_obs, episode_acts).squeeze().detach()
    
    value_s = value_function(episode_obs).squeeze().detach()
    # Handles single state-action pair corner case.
    if(value_s.dim() == 0):
        value_s = value_s.unsqueeze(0) # TODO: I think this can be avoided by using reshape() instead of squeeze()!!!!
    next_value = torch.cat(
        (value_s[1:], torch.tensor([0.0], device=device)),
        dim=0,
    )
    ind_advantages = episode_costs - value_s + gamma*next_value
    
    episode_advantages = stable_reverse_discounting(
        ind_advantages, episode_lambdas * episode_gammas,
    )
    return episode_advantages


@torch.no_grad() # disable gradient tracking in this function.
def get_advantages(
    discriminator: torch.nn.Module,
    value_function: torch.nn.Module,
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    normalize_advantage: bool,
    gamma: float,
    device: torch.DeviceObjType,
):
    """
    Helper function to compute advantages of an episode, returning the tensor
    of concatenated advantages that are normalised according to the input
    parameter `normalize_advantage`.
    """
    all_advantages = []
    for episode in episodes:
        episode_obs, episode_acts, episode_gammas, episode_lambdas = episode
        episode_advantages = get_episode_advantages(
            discriminator=discriminator,
            value_function=value_function,
            episode_obs=episode_obs,
            episode_acts=episode_acts,
            episode_gammas=episode_gammas,
            episode_lambdas=episode_lambdas,
            gamma=gamma,
            device=device,
        )
        all_advantages.append(episode_advantages)
    all_advantages = torch.cat(all_advantages, dim=0)
    if(normalize_advantage):
        all_advantages = (all_advantages - all_advantages.mean()
                            ) / all_advantages.std()
    return all_advantages


@torch.no_grad() # disable gradient tracking in this function.
def get_agent_trajectories(
        env: gym.Env,
        agent_model: torch.nn.Module,
        num_sa_pairs: int,
        horizon: int,
        gamma: float,
        lambd: float,
        device: torch.DeviceObjType,
    ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function queries the agent that is being trained to generate
    trajectories of state-action pairs.

    Parameters:
        env (`gym.Env`): The environment we're using to sample trajectories.
        agent_model (`torch.nn.Module`): The agent model.
        num_sa_pairs (`int`): The number of state-action pairs to sample.
        horizon (`int`): The max number of steps per trajectory.
        gamma (`float`): Discount factor for returns and advantage.
        lambd (`float`): Discount factor for advantage.
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

    # Variables to track trajectory generation.
    agent_obs = []
    agent_actions = []
    agent_gammas = []
    agent_reward = []
    episodes = []
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
            
            if((horizon is not None and 
                num_steps >= horizon) or
                len(agent_obs) >= num_sa_pairs):
                # We have hit the max episode length or generated all the data
                # we needed to generate, finish the episode.
                episode_done = True
            
            # Loop control
            num_steps += 1
        # Episode done, compute all statistics including advantage and return.
        agent_reward.append(episode_reward)

        # First, convert all episode observations to tensors
        episode_obs     = torch.stack(episode_obs, dim=0)
        episode_acts    = torch.stack(episode_acts).to(device).float()
        episode_gammas  = torch.tensor(episode_gammas, device=device).float()
        episode_lambdas = torch.tensor(episode_lambdas, device=device).float()
        agent_gammas.append(episode_gammas)

        episodes.append((
            episode_obs, episode_acts, episode_gammas, episode_lambdas,
        ))
    
    # Convert the lists of tensors to tensors
    agent_obs = torch.stack(agent_obs, dim=0)
    agent_actions = torch.stack(agent_actions).to(device)
    agent_gammas = torch.cat(agent_gammas, dim=0)
    
    # Get relevant logging information
    agent_rwd_mean = np.mean(agent_reward).item()
    
    return (
        agent_rwd_mean,
        agent_obs,
        agent_actions,
        agent_gammas,
        episodes,
    )
