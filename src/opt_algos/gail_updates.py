"""
This file has all the functions necessary to carry out the GAIL-style parameter
updates of the discriminator, value function, and the TRPO policy update.
"""

import torch


def gail_discriminator_update(
    discriminator: torch.nn.Module,
    disc_optim: torch.optim.Optimizer,
    disc_loss_fn,
    expert_obs: torch.Tensor,
    expert_acts: torch.Tensor,
    agent_obs: torch.Tensor,
    agent_acts: torch.Tensor,
    device: torch.DeviceObjType,
) -> torch.Tensor:
    """
    Performs the update on the discriminator for the basline GAIL algorithm.
    Returns the loss for logging purposes.
    """
    # Compute the loss for the discriminator.
    disc_optim.zero_grad()
    expert_scores = discriminator.get_logits(expert_obs, expert_acts)
    agent_scores = discriminator.get_logits(agent_obs, agent_acts)
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
    return disc_loss


