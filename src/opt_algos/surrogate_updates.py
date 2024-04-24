"""
This file contains all the surrogate updates to the GAIL objectives.
"""
# Imports
from omegaconf import DictConfig
import torch
from src.agent_networks.gail_networks import (
    Discriminator,
    PolicyNetwork,
    ValueNetwork,
    Expert,
)
from src.opt_algos.agent_sampling import (
    get_returns,
    get_advantages,
)
from src.opt_algos.gail_updates import (
    KLDivergence,
    TRPOLoss,
)

# Constants
TBS_OPTIMISTIC = "optimistic"
TBS_STD = "standard"


def __compute_disc_surrogate_loss(
    discriminator: Discriminator,
    eta: float,
    expert_obs: torch.Tensor,
    expert_acts: torch.Tensor,
    agent_obs: torch.Tensor,
    agent_acts: torch.Tensor,
    D_t_expert: torch.Tensor,
    D_t_agent: torch.Tensor,
    grad_D_t_expert: torch.Tensor,
    grad_D_t_1_expert: torch.Tensor,
    grad_D_t_agent: torch.Tensor,
    grad_D_t_1_agent: torch.Tensor,
    method: str = TBS_STD,
):
    """
    Helper function to compute the discriminator surrogate loss. See the
    calling function for the form of the loss.
    """
    # Get the current predictions
    D_expert = discriminator(expert_obs, expert_acts)
    D_agent = discriminator(agent_obs, agent_acts)

    # Construct the surrogates
    D_diff_expert = D_expert - D_t_expert
    D_diff_agent = D_agent - D_t_agent
    if(method == TBS_STD):
        expert_inner = grad_D_t_expert * D_diff_expert
        agent_inner = grad_D_t_agent * D_diff_agent
    elif(method == TBS_OPTIMISTIC):
        expert_inner = (2*grad_D_t_expert - grad_D_t_1_expert) * D_diff_expert
        agent_inner = (2*grad_D_t_agent - grad_D_t_1_agent) * D_diff_agent
    
    # Since we're maximising on the discriminator, negate the surrogate.
    expert_surrogate = -expert_inner + (1/(2*eta)) * D_diff_expert.pow(2)
    agent_surrogate = -agent_inner + (1/(2*eta)) * D_diff_agent.pow(2)

    surrogate_loss = torch.cat((expert_surrogate, agent_surrogate)).mean()
    return surrogate_loss


def tbs_discriminator_update(
    discriminator: Discriminator,
    prev_discriminator: Discriminator,
    disc_optim: torch.optim.Optimizer,
    num_inner_loops: int,
    eta: float,
    expert_obs: torch.Tensor,
    expert_acts: torch.Tensor,
    agent_obs: torch.Tensor,
    agent_acts: torch.Tensor,
    method: str = "standard",
    importance_sampling: torch.Tensor = None,
) -> torch.Tensor:
    """
    Update the discriminator using target-based surrogates. The specific method
    used can be selected from <`"standard"`, `"optimistic"`>. The former uses a
    simple update rule: 
        maximise grad(log(D_t(s,a))) * (D - D_t) - 1/(2*eta) * norm(D - D_t)^2
    
    The optimistic update uses a similar form, but adds the previous gradient:
        maximise (2*grad(log(D_t(s,a))) - grad(log(log(D_{t-1}(s,a))))) *
            (D - D_t) - 1/(2*eta) * norm(D - D_t)^2
    
    These updates are performed for both the expert and the agent, the losses
    are then combined and we compute a gradient step for some number of inner
    loops.

    Returns the GAIL-style discriminator loss for comparison purposes and the
    (new) previous discriminator.
    """
    # TODO: Add importance sampling
    # Compute the relevant gradient terms and the D_t() terms.
    with torch.no_grad():
        # Get the current predictions.
        D_t_expert = discriminator(expert_obs, expert_acts)
        D_t_agent  = discriminator(agent_obs, agent_acts)

        # Compute the gradients of the GAIL loss w.r.t. D_t():
        #   agent: log(1 - D_t()).
        #   expert: log(D_t()).
        grad_D_t_expert = 1 / D_t_expert
        grad_D_t_agent  = -1 / (1 - D_t_agent + 1e-8)

        # Get the previous predictions and grads (only for optimistic updates)
        D_t_1_expert = None
        D_t_1_agent = None
        grad_D_t_1_expert = None
        grad_D_t_1_agent = None
        if(method == TBS_OPTIMISTIC):
            D_t_1_expert = prev_discriminator(expert_obs, expert_acts)
            D_t_1_agent  = prev_discriminator(agent_obs, agent_acts)
            grad_D_t_1_expert = 1 / D_t_1_expert
            grad_D_t_1_agent  = -1 / (1 - D_t_1_agent + 1e-8)
    
    # Done with the previous discriminator, can replace it with a new copy.
    # NOTE: This will not pass by reference! A new copy of the params is made!
    if(prev_discriminator is not None):
        prev_discriminator.load_state_dict(discriminator.state_dict())

    for _ in range(num_inner_loops):
        disc_optim.zero_grad()
        loss = __compute_disc_surrogate_loss(
            discriminator=discriminator,
            eta=eta,
            expert_obs=expert_obs,
            expert_acts=expert_acts,
            agent_obs=agent_obs,
            agent_acts=agent_acts,
            D_t_expert=D_t_expert,
            D_t_agent=D_t_agent,
            grad_D_t_expert=grad_D_t_expert,
            grad_D_t_1_expert=grad_D_t_1_expert,
            grad_D_t_agent=grad_D_t_agent,
            grad_D_t_1_agent=grad_D_t_1_agent,
            method=method,
        )
        loss.backward()
        disc_optim.step()
    
    # Updates are done, compute and return the GAIL discriminator loss.
    with torch.no_grad():
        expert_scores = discriminator.get_logits(expert_obs, expert_acts)
        agent_scores = discriminator.get_logits(agent_obs, expert_acts)
        disc_loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        expert_disc_loss = disc_loss_fn(
            expert_scores,
            torch.zeros_like(expert_scores, device=expert_scores.device),
        )
        agent_disc_loss = disc_loss_fn(
            agent_scores,
            torch.ones_like(agent_scores, device=agent_scores.device),
        )
        disc_loss = expert_disc_loss + agent_disc_loss
    return disc_loss


def value_update(
    discriminator: Discriminator,
    value_function: ValueNetwork,
    value_optim: torch.optim.Optimizer,
    value_loss_fn,
    num_inner_loops: int,
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    gamma: float,
    device: torch.DeviceObjType,
):
    """
    Compute the value update a number of times using the discriminator to
    generate agent returns.

    TODO: ADD TBS as an option here!
    """
    old_advantage = None
    for _ in range(num_inner_loops):
        # all_returns = []
        all_advantages = []
        all_obs = []
        value_optim.zero_grad()
        # For each episode, compute the return.
        for episode in episodes:
            episode_obs, episode_acts, episode_gammas, episode_lambdas = episode
            with torch.no_grad():
                # episode_returns = get_returns(
                #     discriminator=discriminator,
                #     episode_obs=episode_obs,
                #     episode_acts=episode_acts,
                #     episode_gammas=episode_gammas,
                # )
                episode_advantages = get_advantages(
                    discriminator=discriminator,
                    value_function=value_function,
                    episode_obs=episode_obs,
                    episode_acts=episode_acts,
                    episode_gammas=episode_gammas,
                    episode_lambdas=episode_lambdas,
                    gamma=gamma,
                    device=device,
                )
            # all_returns.append(episode_returns)
            all_advantages.append(episode_advantages)
            all_obs.append(episode_obs)
        
        all_obs = torch.cat(all_obs, dim=0)
        # all_returns = torch.cat(all_returns, dim=0)
        all_advantages = torch.cat(all_advantages, dim=0)
        if(old_advantage is None):
            old_advantage = all_advantages
        value_s = value_function(all_obs).squeeze()
        all_advantages = value_s.detach() + all_advantages
        total_loss = value_loss_fn(
            value_s,
            all_advantages,
        )
        # TODO: Is gradient clipping actually necessary?
        total_loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm(value_function.parameters(), 10.0)
        # total_loss.backward()
        value_optim.step()
    
    # Finally, compute the loss so we can log it.
    value_function.eval()
    all_advantages = []
    all_obs = []
    with torch.no_grad():
        for episode in episodes:
            episode_obs, episode_acts, episode_gammas, episode_lambdas = episode
            with torch.no_grad():
                # episode_returns = get_returns(
                #     discriminator=discriminator,
                #     episode_obs=episode_obs,
                #     episode_acts=episode_acts,
                #     episode_gammas=episode_gammas,
                # )
                episode_returns = get_advantages(
                    discriminator=discriminator,
                    value_function=value_function,
                    episode_obs=episode_obs,
                    episode_acts=episode_acts,
                    episode_gammas=episode_gammas,
                    episode_lambdas=episode_lambdas,
                    gamma=gamma,
                    device=device,
                )
            all_advantages.append(episode_returns)
            all_obs.append(episode_obs)
        all_obs = torch.cat(all_obs, dim=0)
        all_advantages = torch.cat(all_advantages, dim=0)
        value_s = value_function(all_obs).squeeze()
        total_loss = value_loss_fn(
            value_s,
            all_advantages + value_s,
        )
    new_advantage = all_advantages
    return total_loss, old_advantage, new_advantage


def __compute_policy_surrogate_loss(
    agent_model: PolicyNetwork,
    eta: float,
    agent_obs: torch.Tensor,
    agent_acts: torch.Tensor,
    pi_t: torch.Tensor,
    grad_J_t: torch.Tensor,
    grad_J_t_1: torch.Tensor,
    old_mean: torch.Tensor,
    old_cov: torch.Tensor,
    old_probs: torch.Tensor,
    action_dim: int,
    cfg: DictConfig,
    method: str = TBS_STD,
) -> torch.Tensor:
    """
    Helper function to compute the surrogate loss function for the policy
    network.
    """
    # Get the current estimates and construct the surrogate.
    log_pi_current = agent_model(agent_obs).log_prob(agent_acts)
    pi_current = log_pi_current.exp()
    pi_diff = pi_current - pi_t
    if(method == TBS_STD):
        inner = grad_J_t * pi_diff
    elif(method == TBS_OPTIMISTIC):
        inner = (2*grad_J_t - grad_J_t_1) * pi_diff
    kl_divergence = KLDivergence(
        agent_obs=agent_obs,
        agent_model=agent_model,
        old_mean=old_mean,
        old_cov=old_cov,
        old_probs=old_probs,
        action_dim=action_dim,
        is_discrete=cfg.env.discrete,
    )
    surrogate = -inner + (1 / eta) * kl_divergence()
    surrogate_loss = surrogate.mean()
    return surrogate_loss


def tbs_policy_update(
    discriminator: Discriminator,
    value_function: ValueNetwork,
    agent_model: PolicyNetwork,
    prev_agent_model: PolicyNetwork,
    agent_optim: torch.optim.Optimizer,
    agent_obs: torch.Tensor,
    agent_acts: torch.Tensor,
    agent_gammas: torch.Tensor,
    new_advantages: torch.Tensor,
    old_advantages: torch.Tensor,
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: DictConfig,
    device: torch.DeviceObjType,
):
    """
    Update the agent's policy function using target-based surrogates. The
    specific method used can be selected from <`"standard"`, `"optimistic"`>.
    NOTE: we are heavily basing this target-based surrogate update function off
    of the ones derived in https://arxiv.org/abs/2305.15249, and 
    https://arxiv.org/abs/2108.05828. The novelty of our code is to add an
    optimistic update and to apply this update to GAIL, instead of just raw
    policy gradient methods as was previously done. Note that we are no longer
    minimising the GAIL objective with respect to pi_theta, as one would in a
    GAN, we are instead learning a policy to maximise the expected discounted
    reward (discriminator), as one would in a policy gradient method.
    
    Below, both methods are maximising with respect to the current policy
    function, pi_theta, and updating this function's parameters.
    The "standard" method: 
        maximise grad(J(pi_old)) * (pi_theta - pi_old)
            - (1/eta) * D_KL(pi_theta, pi_old)
    
    The "optimistic" method: 
        maximise (2*grad(J(pi_theta)) - grad(J(pi_old))) * (pi_theta - pi_old)
            - (1/eta) * D_KL(pi_theta, pi_old)
    
    NOTE: The function J(pi) is the expected discounted reward.
        -> It uses importance sampling in the expectation to admit sampling
            from the old distribution.
    """
    # TODO: Add importance sampling
    num_inner_loops = cfg.training_hyperparams.policy_inner_loops
    eta = cfg.training_hyperparams.policy_eta
    method = cfg.training_hyperparams.policy_tbs_method
    ce_lambda = cfg.training_hyperparams.ce_lambda
    gamma = cfg.training_hyperparams.gae_gamma
    normalize_advantage = cfg.training_hyperparams.normalize_advantage

    # Compute the relevant gradient terms and the pi_t terms.
    with torch.no_grad():
        # First, get the up-to-date advantage estimate.
        if(normalize_advantage):
            advantages = (new_advantages - new_advantages.mean()) / new_advantages.std()
            old_advantages = (old_advantages - old_advantages.mean()) / old_advantages.std()
        else:
            advantages = new_advantages

        # Get the current (step t) predictions and gradients w.r.t. pi.
        # TODO: Update gradient here to be the empirical causal entropy -> (agent_gammas / pi)
        distribution_t = agent_model(agent_obs)
        log_pi_t = distribution_t.log_prob(agent_acts)
        pi_t = log_pi_t.exp()
        grad_J_t = advantages / pi_t + ce_lambda*(log_pi_t + 1)

        # Get the previous predictions and grads (only for optimistic updates).
        distribution_t_1 = None
        log_pi_t_1 = None
        pi_t_1 = None
        grad_J_t_1 = None
        if(method == TBS_OPTIMISTIC):
            distribution_t_1 = prev_agent_model(agent_obs)
            log_pi_t_1  = distribution_t_1.log_prob(agent_acts)
            pi_t_1 = log_pi_t_1.exp()
            # TODO: advantages here should be made with the current discriminator but previous value function!!!
            grad_J_t_1  = old_advantages / pi_t_1 + ce_lambda*(log_pi_t_1 + 1)
    
    # Done with the previous agent, can replace it with a new copy.
    # NOTE: This will not pass by reference! A new copy of the params is made!
    if(prev_agent_model is not None):
        prev_agent_model.load_state_dict(agent_model.state_dict())

    # Get variables for the kl divergence.
    old_distribution = agent_model(agent_obs)
    old_mean = old_distribution.mean.detach()
    if(not cfg.env.discrete):
        old_cov_mat = old_distribution.covariance_matrix.sum(-1).detach()
        old_probs = None
    else:
        old_cov_mat = None
        old_probs = old_distribution.probs.detach()
    for _ in range(num_inner_loops):
        agent_optim.zero_grad()
        loss = __compute_policy_surrogate_loss(
            agent_model=agent_model,
            eta=eta,
            agent_obs=agent_obs,
            agent_acts=agent_acts,
            pi_t=pi_t,
            grad_J_t=grad_J_t,
            grad_J_t_1=grad_J_t_1,
            old_mean=old_mean,
            old_cov=old_cov_mat,
            old_probs=old_probs,
            action_dim=agent_model.action_dim,
            cfg=cfg,
            method=method,
        )
        loss.backward()
        agent_optim.step()
    
    # Updates are done, compute and return the GAIL TRPO loss.
    with torch.no_grad():
        trpo_loss_fn = TRPOLoss(
            advantage=advantages,
            agent_acts=agent_acts,
            agent_obs=agent_obs,
            agent_model=agent_model,
            old_log_prob_acts=log_pi_t,
        )
        causal_entropy = ce_lambda *  torch.mean(
            agent_gammas * \
            agent_model(agent_obs).log_prob(agent_acts)
        )
        trpo_loss = trpo_loss_fn() + causal_entropy
    return trpo_loss
