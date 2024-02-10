"""
Proof of concept for rock-paper-scissors toy game.
"""


import torch
from torch.nn import Module
from torch.nn import Parameter
from rock_paper_scissors_env import RPSEnv # Won't use, but it is easy to use.
from tqdm import tqdm


class RPSPlayer(Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.super_complex_neural_network = Parameter(
            torch.randn((3), device=device)
        )
        self.softmax = torch.nn.Softmax(dim=0)
    
    def forward(self, input):
        return self.softmax(self.super_complex_neural_network * input)


def train(num_steps=100000, lamb=10.0, lr=0.01, simultaneous=False):
    """
    Train the simple rock-paper-scissors agents.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_mat = torch.tensor([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]],
        device=device, dtype=torch.float)
    player_1 = RPSPlayer(device)
    player_2 = RPSPlayer(device)
    
    for step in (pbar := tqdm(range(num_steps), unit="Step")):
        player_1.train()
        player_2.train()

        # Get the predictions for each player:
        input = torch.tensor([1], device=device)
        p1_pred = player_1(input)
        p2_pred = player_2(input)

        # Retain the gradient on this non-leaf node
        p1_pred.retain_grad()
        p2_pred.retain_grad()

        # Compute the loss:
        reward = torch.matmul(
            torch.matmul(p1_pred, reward_mat),
            p2_pred,
        )
        loss = reward + (lamb / 2) * (
            torch.linalg.vector_norm(p1_pred - 1/3)**2 - 
            torch.linalg.vector_norm(p2_pred - 1/3)**2
        )

        # Now, compute the relevant gradients
        player_1.zero_grad()
        player_2.zero_grad()
        loss.backward()

        # NOTE: Just changing the parameter updates from += to -= and vice
        # versa, you get totally different optimization behaviour!! Very
        # interesting!
        if(simultaneous or step % 2 == 0):
            p1_grad = p1_pred.grad
            bigP_inv = torch.linalg.pinv(torch.outer(p1_grad, p1_grad)) # Approximate the inverse of the matrix with a pseudo inverse.

            # Update parameters
            player_1.super_complex_neural_network.data -= lr * torch.matmul(
                bigP_inv, player_1.super_complex_neural_network.grad)
        
        if(simultaneous or step % 2 == 1):
            p2_grad = p2_pred.grad
            bigP_inv = torch.linalg.pinv(torch.outer(p2_grad, p2_grad)) # Approximate the inverse of the matrix with a pseudo inverse.

            # Update parameters
            player_2.super_complex_neural_network.data += lr * torch.matmul(
                bigP_inv, player_2.super_complex_neural_network.grad)
        
        if(step % 1000 == 0):
            tqdm.write(f"loss = {loss.item()}")
            tqdm.write(f"p1 policy = {p1_pred[0].item() :.2f}, {p1_pred[1].item() :.2f}, {p1_pred[2].item() :.2f}")
            tqdm.write(f"p2 policy = {p2_pred[0].item() :.2f}, {p2_pred[1].item() :.2f}, {p2_pred[2].item() :.2f}")
    
    return p1_pred, p2_pred, loss.item()


if __name__ == "__main__":
    p1, p2, loss = train()














