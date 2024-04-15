"""
Natural Gradient and Target-based surrogate optimisation of toy games.

TODO: Add random seed setting to get repeatable runs!
"""

import torch
from torch.nn import Module
from torch.nn import Parameter
from functorch import make_functional, jacrev, vmap
import time
from tqdm import tqdm
import wandb


def compute_jacobian_wrt_params(player, input):
    """
    In order to compute the natural gradient update, we need to compute the
    Jacobian with respect to the parameters of the function. This is quite odd
    in normal machine learning since you only really get the gradient with
    respect to your inputs. As such, if we were to define a function which took
    its parameters as inputs, we could compute the jacobian with respect to the
    parameters.
    """
    # Make the neural network a function (i.e., no state, it takes its
    # parameters).
    player_func, player_params = make_functional(player)
    player_param_func = lambda params, x : player_func(params, x)

    # Compute the jacobian with respect to the parameters.
    jacobians = torch.func.jacrev(player_param_func, argnums=(0))(player_params, input) #single sample

    # Collapse the jacobians to be param_size x input_size
    jacobians_param_x_input = torch.cat([
        jac.reshape(-1, input.shape[1]) for jac in jacobians
    ])
    
    # Produces a param_size x param_size matrix
    natural_grad_outer_prod = torch.matmul(
        jacobians_param_x_input,
        jacobians_param_x_input.T,
    )

    # Add small I matrix for stability of the inverse jacobian.
    # NOTE: without the I, training becomes quite unstable.
    natural_grad_outer_prod = natural_grad_outer_prod + torch.eye(
        natural_grad_outer_prod.shape[0], device=input.device) * 1e-3
    
    # Then invert the jacobian.
    natural_grad_outer_prod_inv = torch.pinverse(
        natural_grad_outer_prod,
    )
    return natural_grad_outer_prod_inv

# TODO: implement custom optimiser that can handle this for us. Extremely ugly and annoying code!
@torch.no_grad()
def compute_NG_update(player, inv_outer_prod_jac, lr):
    """
    Natural gradient parameter update.
    """
    # Get all the gradients and concatenate them into a single tensor
    all_grads = torch.cat([
        param.grad.flatten() for param in player.parameters()
    ])
    natural_grad = inv_outer_prod_jac @ all_grads

    idx = 0
    for param in player.parameters():
        param_length = param.numel()
        param_grad = natural_grad[idx : idx + param_length].reshape(param.shape)
        param -= lr * param_grad
        idx += param_length

def train_NG(
        env,
        player_1: Module,
        player_2: Module,
        device: torch.DeviceObjType,
        num_iters=1000,
        base_lr=1e-2,
        lr_schedule=True,
    ):
    """
    Train the agents using the natural gradient algorithm.
    """
    p1_opt = torch.optim.SGD(player_1.parameters(), lr=base_lr)
    p2_opt = torch.optim.SGD(player_2.parameters(), lr=base_lr)
    
    player_1.train()
    player_2.train()
    
    input = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    lr = base_lr
    start_time = time.time()
    for step in (pbar := tqdm(range(num_iters), unit="Step")):
        # Schedule a linearly decreasing learning rate
        if(lr_schedule):
            lr = base_lr * (num_iters - step) / num_iters

        # Get the predictions for each player:
        p1_pred = player_1(input)
        p2_pred = player_2(input)

        # Compute the loss for each player, and update their parameters:
        p1_loss, p2_loss = env.hcc_objective(
            p1_policy=p1_pred, p2_policy=p2_pred)

        inv_outer_prod_jac_1 = compute_jacobian_wrt_params(player_1, input)
        inv_outer_prod_jac_2 = compute_jacobian_wrt_params(player_2, input)

        # Now, compute the updates for player 1
        p1_opt.zero_grad()
        p1_loss.backward()
        compute_NG_update(player_1, inv_outer_prod_jac_1, lr)

        # Now, compute the updates for player 2
        p2_opt.zero_grad()
        p2_loss.backward()
        compute_NG_update(player_2, inv_outer_prod_jac_2, lr)
        
        if(step % 10 == 0):
            tqdm.write(f"loss = {p1_loss.item()}")
            tqdm.write(f"p1 policy = {p1_pred[0,0].item() :.2f}, {p1_pred[0,1].item() :.2f}, {p1_pred[0,2].item() :.2f}")
            tqdm.write(f"p2 policy = {p2_pred[0,0].item() :.2f}, {p2_pred[0,1].item() :.2f}, {p2_pred[0,2].item() :.2f}")
        
        wandb.log({
            "loss": p1_loss.item(),
            "time": time.time() - start_time,
        })
    
    return p1_pred, p2_pred, p1_loss.item()


def compute_surrogate_loss(
        player_current_pred, # w
        player_old_pred,     # w_t
        player_loss_grad,    # grad(L(w_t))
        old_loss,            # L(w_t)
        lr,                  # const
    ):
    """
    The surrogate loss function must have the same gradient as the loss
    function which is then optimised a number of times more than a standard
    loss function.

    Surrogate = L(w_t) + <grad(L(w_t)), w - w_t> + const * norm(w - w_t)^2
    grad(surrogate) = grad(L(w_t))grad(w) + const*2*(w - w_t) = 0
        w_{t+1} = w_t - 1/(const*2) * grad(L(w_t))grad(w)
    
    Where grad(L(w_t))grad(w) = grad(L(w(x_t))) w.r.t. x, which is our
    paramaterisation. Here, w(x_t) is a neural network output that we use in a
    smooth function (loss is at least smooth, if not also convex).
    
    Notice, w_{t+1} is an update to the output of a neural network. How do we
    translate this update from the output space to the parameter space?
        - You can simply do a gradient step on this surrogate loss function
          gradient! x = x_t - const*grad(surrogate). So, just use any standard
          optimiser.
    """
    # Don't want gradients on old models.
    player_old_pred = player_old_pred.detach()
    surrogate_loss = old_loss.detach()

    surrogate_loss += torch.sum(
        player_loss_grad @ (
            player_current_pred - player_old_pred).T
    ) # inner-product
    surrogate_loss += torch.sum(
        (lr / 2) * torch.linalg.vector_norm(
        player_current_pred - player_old_pred) ** 2
    )
    return surrogate_loss


def train_target_based_surrogate(
        env,
        player_1: Module,
        player_2: Module,
        device: torch.DeviceObjType,
        num_iters=1000,
        base_lr=1e-2,
        lr_schedule=True,
    ):
    """
    Train the agents using the target-based surrogates algorithm.
    """
    p1_opt = torch.optim.SGD(player_1.parameters(), lr=base_lr)
    p2_opt = torch.optim.SGD(player_2.parameters(), lr=base_lr)
    
    player_1.train()
    player_2.train()
    
    input = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    lr = base_lr
    start_time = time.time()
    for step in (pbar := tqdm(range(num_iters), unit="Step")):
        # Schedule a linearly decreasing learning rate
        if(lr_schedule):
            lr = base_lr * (num_iters - step) / num_iters

        # Get the predictions for each player:
        p1_pred = player_1(input)
        p2_pred = player_2(input)

        # Compute the game loss for each player:
        p1_loss, p2_loss = env.hcc_objective(
            p1_policy=p1_pred, p2_policy=p2_pred)

        # Get the loss gradients for each player in terms of the network output
        p1_loss_grad = torch.autograd.grad(outputs=p1_loss, inputs=p1_pred, create_graph=True, allow_unused=True)[0].detach()
        p2_loss_grad = torch.autograd.grad(outputs=p2_loss, inputs=p2_pred, create_graph=True, allow_unused=True)[0].detach()
        
        # Now, compute the updates for player 1. Compute the surrogate loss,
        # then, compute the gradient and update model parameters via this loss.
        # breakpoint()
        for _ in range(10):
            p1_opt.zero_grad()
            current_pred = player_1(input)
            p1_surrogate_loss = compute_surrogate_loss(
                player_current_pred=current_pred,
                player_old_pred=p1_pred,
                player_loss_grad=p1_loss_grad,
                old_loss=p1_loss,
                lr=lr,
            )
            p1_surrogate_loss.backward()
            p1_opt.step()

        # Now, compute the updates for player 2
        for _ in range(10):
            p2_opt.zero_grad()
            current_pred = player_2(input)
            p2_surrogate_loss = compute_surrogate_loss(
                player_current_pred=current_pred,
                player_old_pred=p2_pred,
                player_loss_grad=p2_loss_grad,
                old_loss=p2_loss,
                lr=lr,
            )
            p2_surrogate_loss.backward()
            p2_opt.step()
        
        if(step % 10 == 0):
            tqdm.write(f"loss = {p1_loss.item()}")
            tqdm.write(f"p1 policy = {p1_pred[0,0].item() :.2f}, {p1_pred[0,1].item() :.2f}, {p1_pred[0,2].item() :.2f}")
            tqdm.write(f"p2 policy = {p2_pred[0,0].item() :.2f}, {p2_pred[0,1].item() :.2f}, {p2_pred[0,2].item() :.2f}")
        
        wandb.log({
            "loss": p1_loss.item(),
            "time": time.time() - start_time,
        })
    
    return p1_pred, p2_pred, p1_loss.item()













