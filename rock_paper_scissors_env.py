import torch

class RPSEnv():
    """
    Rock-paper-scissors environment for normal RL-style training.
    """
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    ACT_TO_TEXT = {
        ROCK: "rock",
        PAPER: "paper",
        SCISSORS: "scissors",
    }
    TIE = 0
    WIN = 1
    LOSS = -1


    def __init__(self, device):
        """
        Constructor for Rock-Paper-Scissors environment.
        """
        self.device = device
    
    def step(
            self,
            p1_moves: torch.Tensor,
            p2_moves: torch.Tensor,
        ) -> torch.Tensor:
        """
        Take a step (i.e., play a batch of games), and return the results.

        Returns the rewards from p1's perspective.
        """
        # Find when each player wins and when they draw, then assign rewards.
        move_diff = p2_moves - p1_moves

        p1_win = (move_diff == -1).long() + (move_diff == 2).long()
        p2_win = (move_diff == 1).long() + (move_diff == -2).long()

        rewards = torch.zeros_like(p1_win, device=self.device)
        rewards += p1_win * self.WIN + p2_win * self.LOSS

        return rewards