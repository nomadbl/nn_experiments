from self_normalizing_nn import self_normalizing_nn_init

import torch
import torch.nn as nn

class OnChannel(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, inputs: torch.Tensor):
        b, c, h, w = inputs.shape
        inputs = inputs.moveaxis(1, -1).reshape((-1, c))
        output: torch.Tensor = self.layer(inputs)
        _, c = output.shape
        output = output.reshape((b, h, w, c)).moveaxis(-1, 1)
        return output

class OnPatches(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, inputs: torch.Tensor):
        b, c, h, w = inputs.shape
        inputs = inputs.reshape((-1, h*w))
        output: torch.Tensor = self.layer(inputs)
        _, L = output.shape
        assert L==h*w
        output = output.reshape((b, c, h, w))
        return output

class MixerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_patches):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_patches = n_patches
        self.layer = nn.Sequential(
                            OnChannel(self_normalizing_nn_init(nn.Linear(in_dim, out_dim))),
                            nn.SELU(),
                            OnPatches(self_normalizing_nn_init(nn.Linear(n_patches, n_patches))),
                            nn.SELU()
                        )
    
    def forward(self, inputs):
        return self.layer(inputs)