# TheHolyG(RLHF)AIL
Project with Zichu on connections between generative adversarial imitation learning (GAIL) and reinforcement learning from human feedback (RLHF). We want to show how to link the two methods into an encompassing framework and how to design better optimization techniques for this framework. We utilize concepts of hidden convex-concave games.

## How to run the code
We use hydra to manage configurations and WandB to display our metrics during training. These can both be modified in the command-line call of a training run. The following are the two basic methods that we have implemented for the rock-paper-scissors toy problem.

### Basic commands
To run natural gradient-based training on rock-paper-scissors:

`python train.py --config-name=ng_rps.yaml`

To run target-based surrogate training on rock-paper-scissors:

`python train.py --config-name=tbs_rps.yaml`

### Naming WandB runs
The following will allow for wandb run names to be specific directly from the command line. This saves you from manually having to do so in the web application.

`python train.py --config-name=tbs_rps.yaml wandb.run_name="TBS lr\=1e-6"`

### Changing learning hyperparameters
To change any hyperparameter on the command line you can simply add `<high-level name>.<hyperparam>=<value>`. For example, to set the learning rate:

`python train.py --config-name=tbs_rps.yaml training.base_lr=1e-6`
