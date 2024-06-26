"""
General training script where run configurations are specified with Hydra
configs.
"""
import gym
import hydra
import os
import torch
import wandb
from matplotlib.pyplot import set_loglevel
from omegaconf import DictConfig
from src.opt_algos.toy_game_optimisation import train_NG, train_target_based_surrogate
from src.opt_algos.gail_optimisation import train_GAIL


@hydra.main(version_base="1.3", config_path="configs", config_name="tbs_rps.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: None
    """
    # Turn off matplotlib logging.
    set_loglevel("critical")

    checkpoints_dir = "checkpoints"

    wandb_run_name = cfg.wandb.run_name
    del cfg.wandb.run_name # Remove config that wandb doesn't use.
    wandb.init(config=dict(cfg), **cfg.wandb)
    if(wandb_run_name is not None):
        wandb.run.name = wandb_run_name

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if(not cfg.env.gym):
        # If we're not using gym, initialise one of our custom environments.
        del cfg.env.gym
        env = hydra.utils.instantiate(cfg.env, device=device)
    else:
        discrete = cfg.env.discrete
        env = gym.make(cfg.env._target_, render_mode="rgb_array")
        env.reset()
        if(discrete):
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]
        state_dim = len(env.observation_space.high)
        # Can add assert statements here to ensure that action_dim and
        # state_dim match what we have set.

    if(cfg.algo.toy):
        # Optimise a toy game like rock-paper-scissors using the basic
        # implementations of natural gradient or target-based surrogates.
        player_1 = hydra.utils.instantiate(cfg.player_1).to(device)
        player_2 = hydra.utils.instantiate(cfg.player_2).to(device)
        if(cfg.algo.algo == "ng"):
            train_NG(env, player_1, player_2, device, **cfg.training_hyperparams)
        elif(cfg.algo.algo == "tbs"):
            train_target_based_surrogate(env, player_1, player_2, device, **cfg.training_hyperparams)
    elif(cfg.algo.algo == "GAIL"):
        # Expert network from which we sample trajectories.
        expert = hydra.utils.instantiate(cfg.expert_net, state_dim, action_dim, discrete, device).to(device)
        experts_path = os.path.join("experts", cfg.env._target_)
        expert_weights = os.path.join(experts_path, "policy.ckpt")
        expert_state_dict = torch.load(expert_weights, map_location=device)
        expert.pi.load_state_dict(expert_state_dict)

        # Set the checkpoint directory.
        if(wandb_run_name is not None):
            run_and_env = os.path.join(wandb_run_name, cfg.env._target_)
            model_save_dir = os.path.join(checkpoints_dir, run_and_env)
        else:
            model_save_dir = os.path.join(checkpoints_dir, cfg.env._target_)
        
        # Create the directories if need be.
        if(not os.path.exists(model_save_dir)):
            os.makedirs(model_save_dir)

        # Policy we're training.
        policy_net = hydra.utils.instantiate(cfg.policy_net, state_dim, action_dim, discrete, device).to(device)
        # Value (or advantage or quality) network / critic.
        value_net = hydra.utils.instantiate(cfg.value_net, state_dim).to(device)
        # The discriminator in a GAIL setting, this is c(s,a) in the paper
        discriminator = hydra.utils.instantiate(cfg.discriminator_net, state_dim, action_dim, discrete).to(device)
        
        train_GAIL(
            env=env,
            expert=expert,
            agent_model=policy_net,
            value_function=value_net,
            discriminator=discriminator,
            cfg=cfg,
            device=device,
            experts_path=experts_path,
            models_path=model_save_dir,
        )

        # Get rid of the environment, don't need it anymore.
        env.close()

        # Save the trained models
        torch.save(
            policy_net.state_dict(),
            os.path.join(model_save_dir, "policy.pt"),
        )
        torch.save(
            value_net.state_dict(),
            os.path.join(model_save_dir, "value.pt"),
        )
        torch.save(
            discriminator.state_dict(),
            os.path.join(model_save_dir, "discriminator.pt"),
        )
        



if __name__ == "__main__":
    main()


