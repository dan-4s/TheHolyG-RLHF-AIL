"""
This file is taken from https://github.com/hcnoh/gail-pytorch/blob/main/models/nets.py.

We have updated the code to accept model hyperaparameters that allow for
varying model sizes and architectures.

TODO: Add variable model architectures.

NOTE: Discrete means that the action space is discrete, so we have a limited
      number of 0-1 actions.
"""

import torch

from torch.nn import Module, Sequential, Linear, Tanh, Parameter, Embedding
from torch.distributions import Categorical, MultivariateNormal


class PolicyNetwork(Module):
    def __init__(self, state_dim, action_dim, discrete, device=None) -> None:
        super().__init__()

        self.net = Sequential(
            Linear(state_dim, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, action_dim),
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = device
        self.eye = torch.eye(self.action_dim).to(device)

        if not self.discrete:
            self.log_std = Parameter(torch.zeros(action_dim))

    def forward(self, states):
        if self.discrete:
            probs = torch.softmax(self.net(states), dim=-1)
            distb = Categorical(probs)
        else:
            mean = self.net(states)

            std = torch.exp(self.log_std)
            cov_mtx = self.eye * (std ** 2)

            distb = MultivariateNormal(mean, cov_mtx)

        return distb


class ValueNetwork(Module):
    def __init__(self, state_dim) -> None:
        super().__init__()

        self.net = Sequential(
            Linear(state_dim, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 1),
        )

    def forward(self, states):
        return self.net(states)


class Discriminator(Module):
    def __init__(self, state_dim, action_dim, discrete) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if self.discrete:
            self.act_emb = Embedding(
                action_dim, state_dim
            )
            self.net_in_dim = 2 * state_dim
        else:
            self.net_in_dim = state_dim + action_dim

        self.net = Sequential(
            Linear(self.net_in_dim, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 1),
        )

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states: torch.Tensor, actions: torch.Tensor):
        if self.discrete:
            actions = self.act_emb(actions.long())

        sa = torch.cat([states, actions], dim=-1)

        return self.net(sa)


class Expert(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        device=None,
        train_config=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config
        self.device = device

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete, device)

    def get_networks(self):
        return [self.pi]

    def act(self, state: torch.Tensor):
        self.pi.eval() # TODO: This should be in the calling function, not here.
        distb = self.pi(state)

        action = distb.sample().detach().cpu().numpy() # TODO: This should also be in the caller.

        return action
