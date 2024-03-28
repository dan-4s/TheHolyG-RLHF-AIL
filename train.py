"""
General training script where run configurations are specified with Hydra
configs.
"""
import hydra
import torch
import wandb
from omegaconf import DictConfig
from src.opt_algos.toy_game_optimisation import train_NG, train_target_based_surrogate


@hydra.main(version_base="1.3", config_path="configs", config_name="tbs_rps.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: None
    """
    wandb_run_name = cfg.wandb.run_name
    del cfg.wandb.run_name # Remove config that wandb doesn't use.
    wandb.init(config=dict(cfg), **cfg.wandb)
    if(wandb_run_name is not None):
        wandb.run.name = wandb_run_name

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    env = hydra.utils.instantiate(cfg.env, device=device)
    
    player_1 = hydra.utils.instantiate(cfg.player_1).to(device)
    player_2 = hydra.utils.instantiate(cfg.player_2).to(device)

    if(cfg.algo.toy):
        # Optimise a toy game like rock-paper-scissors using the basic
        # implementations of natural gradient or target-based surrogates.
        if(cfg.algo.algo == "ng"):
            train_NG(env, player_1, player_2, device, **cfg.training_hyperparams)
        elif(cfg.algo.algo == "tbs"):
            train_target_based_surrogate(env, player_1, player_2, device, **cfg.training_hyperparams)


if __name__ == "__main__":
    main()


