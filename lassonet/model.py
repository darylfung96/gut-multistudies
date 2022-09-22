from itertools import islice

import torch
from torch import nn
from torch.nn import functional as F

from .prox import inplace_prox, inplace_group_prox, prox


class LassoNet(nn.Module):
    def __init__(self, *dims, groups=None, dropout=None, is_batch_loss=None, condition_latent_size=None):
        """
        first dimension is input
        last dimension is output
        `groups` is a list of list such that `groups[i]`
        contains the indices of the features in the i-th group

        If is_batch_loss is passed, need condition_latent_size so we can do inverse cross entropy on the number of studies
        """
        assert len(dims) > 2
        if groups is not None:
            n_inputs = dims[0]
            all_indices = []
            for g in groups:
                for i in g:
                    all_indices.append(i)
            assert len(all_indices) == n_inputs and set(all_indices) == set(
                range(n_inputs)
            ), f"Groups must be a partition of range(n_inputs={n_inputs})"

        self.groups = groups
        self.is_batch_loss = is_batch_loss

        super().__init__()

        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.skip = nn.Linear(dims[0], dims[-1], bias=False)

        if is_batch_loss:
            assert condition_latent_size is not None, 'need to pass in condition_latent_size if is_batch_loss is passed'
            self.batch_network = nn.Sequential(
                nn.Linear(dims[-2], dims[-2]),
                nn.ReLU(),
                nn.Linear(dims[-2], dims[-2]),
                nn.ReLU(),
                nn.Linear(dims[-2], condition_latent_size)
            )

    def get_encoded_features(self, inp):
        current_layer = inp
        layer_features = []
        for theta in self.layers:
            current_layer = theta(current_layer)
            if theta is not self.layers[-1]:
                if self.dropout is not None:
                    current_layer = self.dropout(current_layer)
                current_layer = F.relu(current_layer)
            layer_features.append(current_layer)
        return layer_features

    def forward(self, inp):
        current_layer = inp
        result = self.skip(inp)
        for theta in self.layers:
            current_layer = theta(current_layer)
            if theta is not self.layers[-1]:
                if self.dropout is not None:
                    current_layer = self.dropout(current_layer)
                current_layer = F.relu(current_layer)

        return result + current_layer

    def get_layer_features(self, inp):
        features = inp
        for layer in self.layers[:-2]:
            features = layer(features)
        return features

    def forward_batch(self, inp):
        features = inp
        for layer in self.layers[:-2]:
            features = layer(features)
        onehot_output = self.batch_network(features)
        return onehot_output

    def prox(self, *, lambda_, lambda_bar=0, M=1):
        if self.groups is None:
            with torch.no_grad():
                inplace_prox(
                    beta=self.skip,
                    theta=self.layers[0],
                    lambda_=lambda_,
                    lambda_bar=lambda_bar,
                    M=M,
                )
        else:
            with torch.no_grad():
                inplace_group_prox(
                    groups=self.groups,
                    beta=self.skip,
                    theta=self.layers[0],
                    lambda_=lambda_,
                    lambda_bar=lambda_bar,
                    M=M,
                )

    def lambda_start(
        self,
        M=1,
        lambda_bar=0,
        factor=2,
    ):
        """Estimate when the model will start to sparsify."""

        def is_sparse(lambda_):
            with torch.no_grad():
                beta = self.skip.weight.data
                theta = self.layers[0].weight.data

                for _ in range(10000):
                    new_beta, theta = prox(
                        beta,
                        theta,
                        lambda_=lambda_,
                        lambda_bar=lambda_bar,
                        M=M,
                    )
                    if torch.abs(beta - new_beta).max() < 1e-5:
                        # print(_)
                        break
                    beta = new_beta
                return (torch.norm(beta, p=2, dim=0) == 0).sum()

        start = 1e-6
        while not is_sparse(factor * start):
            start *= factor
        return start

    def l2_regularization(self):
        """
        L2 regulatization of the MLP without the first layer
        which is bounded by the skip connection
        """
        ans = 0
        for layer in islice(self.layers, 1, None):
            ans += (
                torch.norm(
                    layer.weight.data,
                    p=2,
                )
                ** 2
            )
        return ans

    def l1_regularization_skip(self):
        return torch.norm(self.skip.weight.data, p=2, dim=0).sum()

    def l2_regularization_skip(self):
        return torch.norm(self.skip.weight.data, p=2)

    def input_mask(self):
        with torch.no_grad():
            return torch.norm(self.skip.weight.data, p=2, dim=0) != 0

    def selected_count(self):
        return self.input_mask().sum().item()

    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}