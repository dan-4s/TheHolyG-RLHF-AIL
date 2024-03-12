"""
Proof of concept for rock-paper-scissors toy game.

TODO: Add target-based surrogate optimisation and methods to differentiate between them.
TODO: Add hydra for configuration selection and easy switching between methods.
TODO: Add random seed setting to get repeatable runs!
"""


import torch
from torch.nn import Module
from torch.nn import Parameter
from functorch import make_functional, jacrev, vmap
from rock_paper_scissors_env import RPSEnv
from tqdm import tqdm
import wandb


class TabularRPSPlayer(Module):
    """
    This architecture is too simple, it does not learn anything!
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.super_complex_neural_network = Parameter(
            torch.randn((3), device=device)
        )
        self.softmax = torch.nn.Softmax(dim=0)
    
    def forward(self, input):
        return self.softmax(self.super_complex_neural_network * input)

class MLPRPSPlayer(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPRPSPlayer, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

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

def train(num_steps=1000, base_lr=1e-6):
    """
    Train the simple rock-paper-scissors agents.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = RPSEnv(lambda_reg=1.0, device=device)
    
    player_1 = MLPRPSPlayer(input_size=3, hidden_size=128, output_size=3).to(device)
    player_2 = MLPRPSPlayer(input_size=3, hidden_size=128, output_size=3).to(device)

    p1_opt = torch.optim.SGD(player_1.parameters(), lr=base_lr)
    p2_opt = torch.optim.SGD(player_2.parameters(), lr=base_lr)
    
    player_1.train()
    player_2.train()
    
    input = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    
    for step in (pbar := tqdm(range(num_steps), unit="Step")):
        # Schedule a linearly decreasing learning rate
        lr = base_lr * (num_steps - step) / num_steps

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
            # "": ,
            # "": ,
        })
    
    return p1_pred, p2_pred, p1_loss.item()


if __name__ == "__main__":
    wandb.init(
        project="HCC_NG_vs_TBS",
        tags=["NG_vs_TBS"],
        mode="online",
    )
    p1, p2, loss = train()














