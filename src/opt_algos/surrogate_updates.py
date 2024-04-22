"""
This file contains all the surrogate updates to the GAIL objectives.
"""
# Imports
import copy
import torch


# Constants
TBS_OPTIMISTIC = "optimistic"
TBS_STD = "standard"


def __compute_disc_surrogate_loss(
    discriminator: torch.nn.Module,
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
        expert_inner = grad_D_t_expert.T @ D_diff_expert
        agent_inner = grad_D_t_agent.T @ D_diff_agent
    elif(method == TBS_OPTIMISTIC):
        expert_inner = (2*grad_D_t_expert - grad_D_t_1_expert).T @ D_diff_expert
        agent_inner = (2*grad_D_t_agent - grad_D_t_1_agent).T @ D_diff_agent
    
    expert_surrogate = expert_inner + (1/(2*eta)) * D_diff_expert.norm().pow(2)
    agent_surrogate = agent_inner + (1/(2*eta)) * D_diff_agent.norm().pow(2)

    surrogate_loss = expert_surrogate.mean() + agent_surrogate.mean()
    return surrogate_loss


def tbs_discriminator_update(
    discriminator: torch.nn.Module,
    prev_discriminator: torch.nn.Module,
    disc_optim: torch.optim.Optimizer,
    num_inner_loops: int,
    eta: float,
    expert_obs: torch.Tensor,
    expert_acts: torch.Tensor,
    agent_obs: torch.Tensor,
    agent_acts: torch.Tensor,
    method: str = "standard",
    importance_sampling: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.nn.Module]:
    """
    Update the discriminator using target-based surrogates. The specific method
    used can be selected from <`"standard"`, `"optimistic"`>. The former uses a
    simple update rule: 
        maximise grad(log(D_t(s,a))) * (D - D_t) + 1/(2*eta) * norm(D - D_t)^2
    
    The optimistic update uses a similar form, but adds the previous gradient:
        maximise (2*grad(log(D_t(s,a))) - grad(log(log(D_{t-1}(s,a))))) *
            (D - D_t) + 1/(2*eta) * norm(D - D_t)^2
    
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

