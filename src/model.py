import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class CustomNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(CustomNetwork, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers[:-1])  # remove the last Tanh
        self.apply(init_weights)

    def forward(self, x):
        return self.layers(x)
