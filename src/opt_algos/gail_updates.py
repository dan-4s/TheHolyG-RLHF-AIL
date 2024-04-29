"""
This file has all the functions necessary to carry out the GAIL-style parameter
updates of the discriminator, value function, and the TRPO policy update.
"""
import numpy as np
import torch
from omegaconf import DictConfig
from src.opt_algos.agent_sampling import (
    get_advantages,
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


def gail_policy_update(
    agent_model: torch.nn.Module,
    discriminator: torch.nn.Module,
    value_function: torch.nn.Module,
    agent_optim: torch.optim.Optimizer,
    agent_obs: torch.Tensor,
    agent_actions: torch.Tensor,
    agent_gammas: torch.Tensor,
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: DictConfig,
    device= torch.DeviceObjType,
):
    """
    Compute the loss for the policy network, then update the parameters.
    This is a TRPO step, which includes a natural gradient update of the
    parameters via a line search to satisfy a divergence constraint. The
    implementation is horribly complicated and we should really switch to
    PPO. A basic concept is maintained from the TRPO paper: the
    parameterized distribution yields a mean vector, a covariance matrix,
    and a simplified KL divergence term. In the paper, this is exploited
    to build the Fisher information matrix-vector products instead of
    computing the Hessian-vector product for use in the conjugate
    gradient method. We just compute the Hessian-vector product since we
    can't really assume anything about the distribution to build the FIM.
    
    The loss function is simply the TRPO expected return function:
      Advantage * e^(log(policy) - log(old_policy))
    """
    # NOTE + WARNING: this code is going to be fucking awful.
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
    
    # Get the updated advantages.
    # NOTE: We can just use the episode returns here, we don't NEED to use the
    #   the advantage, this is just what the pytorch-gail implementation did.
    #   Without advantage, we don't need a value function! Can just get the Q-
    #   function directly from the discriminator output!
    agent_advantages = get_advantages(
        discriminator=discriminator,
        value_function=value_function,
        episodes=episodes,
        normalize_advantage=cfg.training_hyperparams.normalize_advantage,
        gamma=cfg.training_hyperparams.gae_gamma,
        device=device,
    )
    
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
        agent_gammas * \
        agent_model(agent_obs).log_prob(agent_actions)
    )
    trpo_loss += causal_entropy.detach()
    grad_causal_entropy = get_flat_grads(
        output=causal_entropy,
        params=agent_model.parameters()
    )
    new_params += grad_causal_entropy

    set_params(agent_model, new_params)
    return trpo_loss
    