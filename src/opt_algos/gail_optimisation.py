"""
This file implements the GAIL optimisation algorithm including the TRPO policy
update step. The code is heavily based on the code from
https://github.com/hcnoh/gail-pytorch/blob/main/models/gail.py.
We update relevant parts and add our own optimisation method.
"""
# Imports
import copy
import gym
import time
import torch
import wandb

from omegaconf import DictConfig
from tqdm import tqdm

from src.opt_algos.expert_sampling import get_expert_trajectories
from src.opt_algos.agent_sampling import get_agent_trajectories
from src.agent_networks.gail_networks import (
    Discriminator,
    PolicyNetwork,
    Expert,
    ValueNetwork
)
from src.opt_algos.gail_updates import (
    gail_discriminator_update,
    gail_policy_update,
)
from src.opt_algos.surrogate_updates import (
    tbs_discriminator_update,
    value_update,
    tbs_policy_update,
)

# Constants
GAIL_BASE = "GAIL"
GAIL_TBS  = "TBS"
from src.opt_algos.surrogate_updates import (
    TBS_STD,
    TBS_OPTIMISTIC,
)


def train_GAIL(
        env: gym.Env,
        expert: Expert,
        agent_model: PolicyNetwork,
        value_function: ValueNetwork,
        discriminator: Discriminator,
        cfg: DictConfig,
        device: torch.DeviceObjType,
        experts_path: str = None,
    ) -> None:
    """
    Training function for generative adversarial imitation learning (GAIL).

    Parameters:
        env (`gym.Env`): The environment we're using to sample trajectories.
        expert (`torch.nn.Module`): The expert model.
        agent_model (`torch.nn.Module`): The agent model.
        value_function (`torch.nn.Module`): The value function model (acts as a
            critic function in actor-critic).
        discriminator (`torch.nn.Module`): The discriminator model (acts as a
            Q-function).
        cfg ('omegaconf.DictConfig'): Configurations for the expert and agent.
        device (`torch.device`): The device to put all data on.
        experts_path (`str`): [Optional] Path to experts directory.

    Returns:
        No output.
    """
    start_time = time.time()

    # First, either collect or gather the expert trajectories.
    # TODO: Add trajectory saving / loading.
    expert_reward_mean, expert_obs, expert_actions = get_expert_trajectories(
        env=env,
        expert=expert,
        num_sa_pairs=cfg.expert_hyperparams.num_steps_per_iter,
        horizon=cfg.expert_hyperparams.horizon,
        device=device,
        render_gif=cfg.expert_hyperparams.render_gif,
        gif_path=experts_path,
    )
    print(f"Expert reward mean: {expert_reward_mean}")
    wandb.log({
        "Expert Reward Mean": expert_reward_mean,
        "Time (minutes)": (time.time() - start_time) / 60,
    })

    # Training loop: for num_iters, train the model with successive updates to:
    #   1. The discriminator: minimise log(D(agent)) + log(1-D(expert))
    #   2. The value function: MSE loss between advantages and value prediction
    #   3. The policy: TRPO update
    num_iters = cfg.training_hyperparams.num_iters
    disc_loss_fn = torch.nn.BCEWithLogitsLoss()
    disc_opt_method = cfg.training_hyperparams.disc_opt_method
    disc_tbs_method = cfg.training_hyperparams.disc_tbs_method
    disc_inner_loops = cfg.training_hyperparams.disc_inner_loops
    disc_eta = cfg.training_hyperparams.disc_eta
    prev_discriminator = None
    if(disc_opt_method == GAIL_TBS and disc_tbs_method == TBS_OPTIMISTIC):
        prev_discriminator = copy.deepcopy(discriminator)
    disc_optim = torch.optim.AdamW(
        discriminator.parameters(),
        lr=cfg.training_hyperparams.lr,
    )

    value_loss_fn = torch.nn.MSELoss()
    value_inner_loops = cfg.training_hyperparams.value_inner_loops
    value_optim = torch.optim.AdamW(
        value_function.parameters(),
        lr=cfg.training_hyperparams.lr,
    )

    agent_opt_method = cfg.training_hyperparams.policy_opt_method
    agent_tbs_method = cfg.training_hyperparams.policy_tbs_method
    agent_inner_loops = cfg.training_hyperparams.policy_inner_loops
    agent_eta = cfg.training_hyperparams.policy_eta
    prev_agent = None
    if(agent_opt_method == GAIL_TBS and agent_tbs_method == TBS_OPTIMISTIC):
        prev_agent = copy.deepcopy(agent_model)
    agent_optim = torch.optim.AdamW(
        agent_model.parameters(),
        lr=cfg.training_hyperparams.lr,
    )
    for step in (pbar := tqdm(range(num_iters), unit="Step")):
        out = get_agent_trajectories(
            env=env,
            agent_model=agent_model,
            value_function=value_function,
            discriminator=discriminator,
            num_sa_pairs=cfg.training_hyperparams.num_steps_per_iter,
            horizon=cfg.training_hyperparams.horizon,
            gamma=cfg.training_hyperparams.gae_gamma,
            lambd=cfg.training_hyperparams.gae_lambda,
            normalize_advantage=cfg.training_hyperparams.normalize_advantage,
            device=device,
        )
        (
            agent_reward_mean,
            agent_obs,
            agent_actions,
            old_agent_returns,    # OLD estimates, useful for TBS-style training.
            old_agent_advantages, # OLD estimates, useful for TBS-style training.
            agent_gammas,
            episodes, # Used to generate new return and advantage estimates.
        ) = out
        tqdm.write(f"STEP {step}: Agent reward mean {agent_reward_mean}")

        # Compute the loss and update for the discriminator.
        discriminator.train()
        if(disc_opt_method == GAIL_BASE):
            disc_loss = gail_discriminator_update(
                discriminator=discriminator,
                disc_optim=disc_optim,
                disc_loss_fn=disc_loss_fn,
                expert_obs=expert_obs,
                expert_acts=expert_actions,
                agent_obs=agent_obs,
                agent_acts=agent_actions,
                device=device,
            )
        elif(disc_opt_method == GAIL_TBS):
            disc_loss = tbs_discriminator_update(
                discriminator=discriminator,
                prev_discriminator=prev_discriminator,
                disc_optim=disc_optim,
                num_inner_loops=disc_inner_loops,
                eta=disc_eta,
                expert_obs=expert_obs,
                expert_acts=expert_actions,
                agent_obs=agent_obs,
                agent_acts=agent_actions,
                method=disc_tbs_method,
                importance_sampling=None,
            )
        discriminator.eval()

        # Compute the loss for the value function, then compute the parameter
        # update. The value loss function is the MSE between the agent value
        # estimate and the discounted returns that we approximated in the agent
        # sampling function (discounted D(s,a) for each state in a trajectory).
        # NOTE: There are a few different ways to do this:
        #   1. From https://github.com/hcnoh/gail-pytorch/, use the conjugate
        #       gradient method to compute the update direction, then take a
        #       step in this direction. Specifically, you compute the Hessian-
        #       vector product of your constraint (you don't want the value to
        #       change drastically, so use an MSE loss of model outputs). The
        #       gradient is provided to the algorithm to find the direction.
        #   2. From https://github.com/ikostrikov/pytorch-trpo/blob/master/main.py,
        #       uses the `scipy.optimize.fmin_l_bfgs_b` method to minimise the
        #       value loss function (MSE loss with weight decay). L-BFGS is a
        #       limited-memory algorithm which approximates the Newton (i.e.,
        #       the hessian-gradient product) descent direction. However,
        #       unlike 1., it does not explicitly form the Hessian. This is the
        #       same method that the original TRPO paper used.
        #   3. We can simply do gradient descent on the loss function... This
        #       is the easiest method by far.
        #
        # For the sake of simplicity, we implement 3, with the knowledge that
        # this may not be the best for convergence.
        #
        # NOTE: The discriminator has since been updated, we need to recompute
        #   the returns!
        value_function.train()
        value_loss = value_update(
            discriminator=discriminator,
            value_function=value_function,
            value_optim=value_optim,
            value_loss_fn=value_loss_fn,
            num_inner_loops=value_inner_loops,
            episodes=episodes,
        )
        value_function.eval()

        # TODO: policy update step
        agent_model.train()
        if(agent_opt_method == GAIL_BASE):
            policy_loss = gail_policy_update(
                agent_model=agent_model,
                discriminator=discriminator,
                value_function=value_function,
                agent_optim=agent_optim,
                agent_obs=agent_obs,
                agent_actions=agent_actions,
                agent_gammas=agent_gammas,
                episodes=episodes,
                cfg=cfg,
                device=device,
            )
        else:
            policy_loss = tbs_policy_update(
                discriminator=discriminator,
                value_function=value_function,
                agent_model=agent_model,
                prev_agent_model=prev_agent,
                agent_optim=agent_optim,
                agent_obs=agent_obs,
                agent_acts=agent_actions,
                agent_gammas=agent_gammas,
                episodes=episodes,
                cfg=cfg,
                device=device,
            )
        agent_model.eval()

        # breakpoint()
        # Log relevant values
        wandb.log({
            "Agent Reward Mean": agent_reward_mean,
            "Time (minutes)": (time.time() - start_time) / 60,
            "TRPO LOSS": policy_loss.item(),
            "VALUE LOSS": value_loss.item(),
            "DISCRIMINATOR LOSS": disc_loss.item(),
        })
