"""
This file implements the expert trajectory sampling algorithm for GAIL.
Specifically, we take the code from 
https://github.com/hcnoh/gail-pytorch/blob/main/models/gail.py,
updating it for better performance.

TODO: investigate vectorised environments.
TODO: the expert is only sampling num_sa_pairs over a single trajectory
    potentially, I don't think this is intended.. it makes the expert
    dataset quite small and might make the GAIL optimisation a bit more
    difficult than it needs to be.. might just need more data.
"""

import gym
import numpy as np
import os
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation


@torch.no_grad() # disable gradient tracking in this function.
def get_expert_trajectories(
        env: gym.Env,
        expert: torch.nn.Module,
        num_sa_pairs: int,
        horizon: int,
        device: torch.DeviceObjType,
        render_gif: bool = False,
        gif_path: str = None,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    This function queries the expert model to generate trajectories of
    state-action pairs.

    Parameters:
        env (`gym.Env`): The environment we're using to sample trajectories.
        expert (`torch.nn.Module`): The expert model.
        num_sa_pairs (`int`): The number of state-action pairs to sample.
        horizon (`int`): The max number of steps per trajectory.
        device (`torch.device`): The device to put all data on.
        render_gif (`bool`): Whether to output a gif of the expert actions.
        gif_path (`str`): Where to save the gif.

    Returns:
        `tuple[float, torch.Tensor, torch.Tensor]`: the expert reward mean,
            state and action tensors.
    """
    expert.eval()

    # Variables to track trajectory generation.
    num_gen_traj = 0
    expert_obs = []
    expert_actions = []
    expert_reward = []
    while len(expert_obs) < num_sa_pairs:
        # Run trajectories until we have a sufficient amount of data.
        gif_frames = []
        obs, _ = env.reset()
        if(render_gif):
            gif_frames.append(env.render())
        episode_done = False
        episode_reward = 0
        num_steps = 0
        while not episode_done:
            # Run an episode to generate a trajectory.
            obs = torch.tensor(obs, device=device).float()
            expert_action = expert.act(obs)
            expert_obs.append(obs)
            expert_actions.append(torch.tensor(expert_action))

            obs, reward, episode_done, _, _ = env.step(expert_action)
            episode_reward += reward

            if(render_gif):
                gif_frames.append(env.render())
            
            if(horizon is not None and 
                num_steps >= horizon or
                len(expert_obs) >= num_sa_pairs):
                # We have hit the max episode length or generated all the data
                # we needed to generate, finish the episode.
                episode_done = True
            
            # Loop control
            num_steps += 1
        expert_reward.append(episode_reward)
        
        if(render_gif):
            # Create the figure and axes objects
            fig, ax = plt.subplots()

            # Set the initial image
            im = ax.imshow(gif_frames[0], animated=True)

            def update(i):
                im.set_array(gif_frames[i])
                return im,

            # Create the animation object
            animation_fig = animation.FuncAnimation(fig, update, frames=len(gif_frames), interval=40, blit=True,repeat_delay=10,)

            # Show the animation
            plt.show()
            plt.title(f"Expert Trajectory #{num_gen_traj+1}")

            animation_fig.save(os.path.join(gif_path, f"trajectory_{num_gen_traj+1}.gif"))
        num_gen_traj += 1
    
    # Get relevant logging information
    exp_rwd_mean = np.mean(expert_reward).item()

    # Convert data to tensors.
    t_obs = torch.stack(expert_obs, dim=0)
    t_acts = torch.tensor(expert_actions, device=device).float()

    return exp_rwd_mean, t_obs, t_acts
