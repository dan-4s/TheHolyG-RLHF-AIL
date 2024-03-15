"""
General training script where run configurations are specified with Hydra
configs.
"""
import hydra
import wandb
from omegaconf import DictConfig
from rock_paper_scissors import train_NG, train_target_based_surrogate


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
    
    if(cfg.algo.algo == "ng"):
        train_NG(**cfg.training_hyperparams)
    elif(cfg.algo.algo == "tbs"):
        train_target_based_surrogate(**cfg.training_hyperparams)


if __name__ == "__main__":
    main()


