import torch

class RPSEnv():
    """
    Toy rock-paper-scissors environment in a hidden convex-concave game style.
    """
    def __init__(self, lambda_reg: float, device: torch.DeviceObjType):
        """
        Constructor for Rock-Paper-Scissors environment.
        """
        self.lambda_reg = lambda_reg
        self.device = device
        self.reward_mat = torch.tensor([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]],
            device=device, dtype=torch.float)
    
    def __hcc_loss(self, p1_policy, p2_policy):
        # We have the policies of each player, compute the loss
        reward = torch.sum(p1_policy @ self.reward_mat @ p2_policy.T)
        loss = reward + (self.lambda_reg / 2) * (
            torch.linalg.vector_norm(p1_policy - 1/3)**2 - 
            torch.linalg.vector_norm(p2_policy - 1/3)**2
        )
        return loss
    
    def hcc_objective(
        self,
        p1_policy: torch.Tensor,
        p2_policy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the hidden convex-concave game objective given the player
        policies. This method returns two losses:
            1) The loss for the first player with player 2's gradient's
                detached.
            2) The loss for the second player with player 1's gradient's
                detached.
        """
        # Player 1 is being learned, so update player 1, and freeze player 2.
        loss_p1 = self.__hcc_loss(p1_policy, p2_policy.detach())
        
        # Freeze p1, learn p2. Loss is negated since p2 is maximising.
        loss_p2 = -self.__hcc_loss(p1_policy.detach(), p2_policy)
        return loss_p1, loss_p2
        
        
        