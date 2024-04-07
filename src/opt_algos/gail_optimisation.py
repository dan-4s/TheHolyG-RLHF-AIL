"""
This file implements the GAIL optimisation algorithm including the TRPO policy
update step. The code is heavily based on the code from
https://github.com/hcnoh/gail-pytorch/blob/main/models/gail.py.
We update relevant parts and add our own optimisation method.
"""

import gym
import numpy as np
import os
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


class TRPOLoss():
    """
    Class to handle loss function with state.
    """
    def __init__(
        self,
        advantage: torch.Tensor,
        agent_acts: torch.Tensor,
        agent_obs: torch.Tensor,
        agent_model: torch.nn.Module,
        old_log_prob_acts: torch.Tensor,
    ) ->  None:
        self.advantage = advantage
        self.agent_acts = agent_acts
        self.agent_obs = agent_obs
        self.agent_model = agent_model
        self.old_log_prob_acts = old_log_prob_acts
    
    def __call__(self):
        """
        Returns the TRPO loss from the saved variables.
        """
        current_log_prob_acts = self.agent_model(
            self.agent_obs).log_prob(self.agent_acts)
        diff_log_probs = current_log_prob_acts - self.old_log_prob_acts
        return torch.mean(self.advantage * torch.exp(diff_log_probs))


class KLDivergence():
    """
    Class to handle KL Divergence function with state.
    """
    def __init__(
        self,
        agent_obs: torch.Tensor,
        agent_model: torch.nn.Module,
        old_mean: torch.Tensor,
        old_cov: torch.Tensor,
        old_probs: torch.Tensor,
        action_dim: int,
        is_discrete: bool,
    ) -> None:
        self.agent_obs = agent_obs
        self.agent_model = agent_model
        self.old_mean = old_mean
        self.old_cov = old_cov
        self.old_probs = old_probs
        self.action_dim = action_dim
        self.is_dicrete = is_discrete
    
    def __call__(self):
        """
        Returns the distribution-based KL divergence between the distributions
        induced by the old / new policy networks.
        """
        distb = self.agent_model(self.agent_obs)

        if(self.is_discrete):
            p = distb.probs
            return (self.old_probs * (
                torch.log(self.old_probs) - torch.log(p)
                )).sum(-1).mean()
        else:
            mean = distb.mean
            cov = distb.covariance_matrix.sum(-1)
            return (0.5) * (
                    (self.old_cov / cov).sum(-1)
                    + (((self.old_mean - mean) ** 2) / cov).sum(-1)
                    - self.action_dim
                    + torch.log(cov).sum(-1)
                    - torch.log(self.old_cov).sum(-1)
                ).mean()


def get_flat_grads(output, params):
    """
    Taking the output of a function and the parameters of a network, find the
    gradients of the function output with respect to the parameters of the
    network.
    """
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=params,
        create_graph=True,
    )
    return torch.cat([grads.view(-1) for grads in gradients])


class HessianVectorProduct():
    """
    Class to function with state.
    """
    def __init__(
        self,
        agent_model: torch.nn.Module,
        old_grad: torch.Tensor,
        damping: float,
    ) -> None:
        self.agent_model = agent_model
        self.old_grad = old_grad
        self.damping = damping
    
    def __call__(self, vector: torch.Tensor):
        """
        Compute the Hessian-vector product between the old gradient and the
        input vector.
        """
        hessian = get_flat_grads(
            output=torch.dot(self.old_grad, vector),
            params=self.agent_model.parameters(),
        ).detach()
        return hessian + self.damping * vector


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
    #   2. The value function: 
    #   3. The policy: 
    num_iters = cfg.training_hyperparams.num_iters
    disc_loss_fn = torch.nn.BCEWithLogitsLoss()
    value_loss_fn = torch.nn.MSELoss()
    learning_rate = cfg.training_hyperparams.lr
    disc_optim = torch.optim.AdamW(
        discriminator.parameters(),
        lr=learning_rate,
    )
    value_optim = torch.optim.AdamW(
        value_function.parameters(),
        lr=learning_rate,
    )
    # This is for convenience, we don't actually use the agent's optimiser!
    agent_optim = torch.optim.SGD(
        agent_model.parameters(),
        lr=learning_rate,
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
            agent_returns,
            agent_advantages,
            agent_gammas
        ) = out
        tqdm.write(f"STEP {step}: Agent reward mean {agent_reward_mean}")

        # Compute the loss for the discriminator.
        discriminator.train()
        disc_optim.zero_grad()
        expert_scores = discriminator.get_logits(expert_obs, expert_actions)
        agent_scores = discriminator.get_logits(agent_obs, agent_actions)
        expert_disc_loss = disc_loss_fn(
            expert_scores,
            torch.zeros_like(expert_scores, device=device),
        )
        agent_disc_loss = disc_loss_fn(
            agent_scores,
            torch.ones_like(agent_scores, device=device),
        )
        disc_loss = expert_disc_loss + agent_disc_loss
        disc_loss.backward()
        disc_optim.step()

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
        value_function.train()
        value_optim.zero_grad()
        agent_value_estimate = value_function(agent_obs).squeeze()
        value_loss = value_loss_fn(agent_value_estimate, agent_returns)
        value_loss.backward()
        value_optim.step()

        # Compute the loss for the policy network, then update the parameters.
        # This is a TRPO step, which includes a natural gradient update of the
        # parameters via a line search to satisfy a divergence constraint. The
        # implementation is horribly complicated and we should really switch to
        # PPO. A basic concept is maintained from the TRPO paper: the
        # parameterized distribution yields a mean vector, a covariance matrix,
        # and a simplified KL divergence term. In the paper, this is exploited
        # to build the Fisher information matrix-vector products instead of
        # computing the Hessian-vector product for use in the conjugate
        # gradient method. We just compute the Hessian-vector product since we
        # can't really assume anything about the distribution to build the FIM.
        #
        # The loss function is simply the TRPO expected return function:
        #   Advantage * e^(log(policy) - log(old_policy))
        #
        # NOTE + WARNING: this code is going to be fucking awful.
        agent_model.train()
        agent_optim.zero_grad()
        old_distribution = agent_model(agent_obs)
        old_mean = old_distribution.mean.detach()
        old_probs = old_distribution.probs.detach()
        old_log_prob_acts = old_distribution.log_prob(agent_actions).detach()
        old_cov_mat = old_distribution.covariance_matrix.sum(-1).detach()

        # Set up the TRPO loss and KL Divergence.
        trpo_loss_fn = TRPOLoss(
            advantage=agent_advantages,
            agent_acts=agent_actions,
            agent_obs=agent_obs,
            agent_model=agent_model,
            old_log_prob_acts=old_log_prob_acts,
        )
        kl_divergence_fn = KLDivergence(
            agent_obs=agent_obs,
            agent_model=agent_model,
            old_mean=old_mean,
            old_cov=old_cov_mat,
            old_probs=old_probs,
            action_dim=cfg.env.action_dim,
            is_discrete=cfg.env.discrete,
        )

        # Set up the Hessian-vector product of the KL divergence.
        #   First, get the gradient of the KL w.r.t. the current model params.
        flat_old_kl_grads = get_flat_grads(
            output=kl_divergence_fn(),
            params=agent_model.parameters(),
        )
        hessian_vector_prod_fn = HessianVectorProduct(
            agent_model=agent_model,
            old_grad=flat_old_kl_grads,
            damping=cfg.training_hyperparams.cg_damping,
        )

        # Now, we're ready to use the conjugate gradient algorithm!
        # TODO! 







        
        



