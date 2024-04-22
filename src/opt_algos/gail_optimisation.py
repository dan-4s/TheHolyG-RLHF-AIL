"""
This file implements the GAIL optimisation algorithm including the TRPO policy
update step. The code is heavily based on the code from
https://github.com/hcnoh/gail-pytorch/blob/main/models/gail.py.
We update relevant parts and add our own optimisation method.
"""
# Imports
import copy
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
from src.opt_algos.gail_updates import (
    gail_discriminator_update,
)
from src.opt_algos.surrogate_updates import (
    tbs_discriminator_update,
)

# Constants
GAIL_BASE = "GAIL"
GAIL_TBS  = "TBS"
from src.opt_algos.surrogate_updates import (
    TBS_STD,
    TBS_OPTIMISTIC,
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
        self.is_discrete = is_discrete
    
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


def conjugate_gradient(
    hessian_vector_prod_fn: HessianVectorProduct,
    loss_gradient: torch.Tensor,
) -> torch.Tensor:
    """
    Using the conjugate gradient method, computes the descent direction by
    sucessively minimising the residual between the Hessian-vector product term
    and the gradient. Once the residual is sufficiently small, we return the
    vector which achieved the small residual.

    Estimate x such that Ax = b, for a known A and b.
        - r1 = b - Ax1
        - x2 = x + alpha * r1
        - r2 = r1 - alpha * Ar1 -> This is a descent step on the residual!!!
        - ...
    """
    # Code taken from the GAIL implementation.
    x = torch.zeros_like(loss_gradient)
    r = loss_gradient - hessian_vector_prod_fn(x)
    p = r
    rsold = r.norm() ** 2

    for _ in range(10):
        Ap = hessian_vector_prod_fn(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if(torch.sqrt(rsnew) < 1e-10):
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def get_flat_params(net):
    return torch.cat([param.view(-1) for param in net.parameters()])


def set_params(net, new_flat_params):
    start_idx = 0
    for param in net.parameters():
        end_idx = start_idx + np.prod(list(param.shape))
        param.data = torch.reshape(
            new_flat_params[start_idx:end_idx], param.shape
        )
        start_idx = end_idx


def line_search(
    max_kl_constraint: float,
    loss_gradient: torch.Tensor,
    descent_dir: torch.Tensor,
    hessian_descent_dir: torch.Tensor,
    kl_div_fn: KLDivergence,
    loss_fn: TRPOLoss,
    model: torch.nn.Module,
):
    """
    Do a line search to find a feasible point where the KL constraint is
    satisfied and the loss has improved by a satisfactory amount.
    """
    if(hessian_descent_dir.isnan().sum() > 0):
        print("HESSIAN NAN!!!")
    # breakpoint()
    old_params = get_flat_params(model)
    old_loss = loss_fn().detach()

    # Set the initial step size
    beta = torch.sqrt(
        2 * max_kl_constraint / torch.dot(descent_dir, hessian_descent_dir))

    for _ in range(10):
        new_params = old_params + beta * descent_dir
        set_params(model, new_params)
        new_kl_div = kl_div_fn().detach()
        new_loss = loss_fn().detach()

        improvement = new_loss - old_loss
        approx_improvement = torch.dot(loss_gradient, beta*descent_dir)
        improvement_ratio = improvement / approx_improvement
        if(improvement_ratio > 0.1 and
           improvement > 0 and
           new_kl_div < max_kl_constraint):
            return new_params
        
        # Exponentially decay the step size, beta.
        beta *= 0.5
    
    print("No feasible solution found in line search")
    return old_params


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
    value_optim = torch.optim.AdamW(
        value_function.parameters(),
        lr=cfg.training_hyperparams.lr,
    )

    # This is for convenience, we don't actually use the agent's optimiser!
    agent_optim = torch.optim.SGD(
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
            agent_returns,
            agent_advantages,
            agent_gammas
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
        old_log_prob_acts = old_distribution.log_prob(agent_actions).detach()
        if(not cfg.env.discrete):
            old_cov_mat = old_distribution.covariance_matrix.sum(-1).detach()
            old_probs = None
        else:
            old_cov_mat = None
            old_probs = old_distribution.probs.detach()

        # Set up the TRPO loss and KL Divergence.
        trpo_loss_fn = TRPOLoss(
            advantage=agent_advantages,
            agent_acts=agent_actions,
            agent_obs=agent_obs,
            agent_model=agent_model,
            old_log_prob_acts=old_log_prob_acts,
        )
        # if(trpo_loss_fn().isnan().item()):
        #     breakpoint()
        kl_divergence_fn = KLDivergence(
            agent_obs=agent_obs,
            agent_model=agent_model,
            old_mean=old_mean,
            old_cov=old_cov_mat,
            old_probs=old_probs,
            action_dim=agent_model.action_dim,
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

        # Now, we're ready to use the conjugate gradient algorithm! The steps
        # are:
        #   1. Compute the gradient of your loss function.
        #   2. Using this gradient, compute the descent direction using the
        #       conjugate gradient algorithm -> find x s.t. Hx = gradient!
        #       Where x is the descent direction. and H is the Hessian of the
        #       KL divergence. -> notice: x = H^-1 * gradient, this is Newton's
        #       method!!!
        #   3. Search along this descent direction for a parameter update that
        #       satisfies the TRPO constraints: small KL divergence, and a
        #       monotonically increasing loss function (reward)!
        trpo_loss = trpo_loss_fn()
        
        loss_gradient = get_flat_grads(
            output=trpo_loss,
            params=agent_model.parameters(),
        ).detach() # Detach since we don't want to compute gradients on this.
        
        descent_dir = conjugate_gradient(
            hessian_vector_prod_fn, loss_gradient).detach()
        hessian_descent_dir = hessian_vector_prod_fn(descent_dir).detach()

        # Now do a line search.
        new_params = line_search(
            max_kl_constraint=cfg.training_hyperparams.max_kl,
            loss_gradient=loss_gradient,
            descent_dir=descent_dir,
            hessian_descent_dir=hessian_descent_dir,
            kl_div_fn=kl_divergence_fn,
            loss_fn=trpo_loss_fn,
            model=agent_model,
        )

        # Now, update the parameters according to the causal entropy
        # regulariser.
        # NOTE: doing this after the line search means we are updating the
        # params according to the causal entropy after we updated them for the
        # line search... This may not be ideal.
        causal_entropy = cfg.training_hyperparams.ce_lambda *  torch.mean(
            (-1) * agent_gammas *\
            agent_model(agent_obs).log_prob(agent_actions)
        )
        trpo_loss += causal_entropy.detach()
        grad_causal_entropy = get_flat_grads(
            output=causal_entropy,
            params=agent_model.parameters()
        )
        new_params += grad_causal_entropy

        set_params(agent_model, new_params)

        # breakpoint()
        # Log relevant values
        wandb.log({
            "Agent Reward Mean": agent_reward_mean,
            "Time (minutes)": (time.time() - start_time) / 60,
            "TRPO LOSS": trpo_loss.item(),
            "VALUE LOSS": value_loss.item(),
            "DISCRIMINATOR LOSS": disc_loss.item(),
        })
