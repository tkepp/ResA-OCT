#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Generalizable SIREN by latent code modularization. Inspired by NISF.

"""
import torch
from torch import nn
from typing import Tuple
import numpy as np




class ModulatedSiren(nn.Module):
    def __init__(self, coord_size: int, embed_size: int, hidden_size:int, num_hidden_layers:int, output_size:int,  **kwargs):
        super(ModulatedSiren, self).__init__()

        self.omega_0 = kwargs.get("siren_omega_0", 30.0)
        self.siren_init = kwargs.get("siren_init", None)

        net_layer_list = [nn.Linear(coord_size, hidden_size)]
        mod_layer_list = []
        for i in range(num_hidden_layers - 1):
            net_layer_list.append(nn.Linear(hidden_size, hidden_size))
            mod_layer_list.append(nn.Linear(embed_size, hidden_size * 2))
        self.hid_layers = nn.ModuleList(net_layer_list)
        self.mod_layers = nn.ModuleList(mod_layer_list)
        self.last_layer = nn.Linear(hidden_size, output_size)


        self.out_size = hidden_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.reset_params(init_value=self.siren_init)

    @staticmethod
    def weights_init(model, init_value = None, omega_0=30.0):

        with torch.no_grad():
            if isinstance(model, nn.Linear):
                if init_value is None:
                    hidden_ch = max(model.weight.shape[0], model.weight.shape[1])
                    val = np.sqrt(6 / hidden_ch) / omega_0 if model.weight.shape[1] > 4 else 1 / model.weight.shape[1]
                else:
                    val = init_value if model.weight.shape[1] > 4 else 1 / model.weight.shape[1]
                torch.nn.init.uniform_(model.weight, -val, val)
                if model.bias is not None:
                    torch.nn.init.zeros_(model.bias)

    def reset_params(self, init_value):
        for i, module in enumerate(self.modules()):
            self.weights_init(module, init_value=init_value, omega_0=self.omega_0)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        coords, h = x
        out = torch.sin(self.omega_0 * self.hid_layers[0](coords))

        for i in range(1, len(self.hid_layers) - 1):
            phi_psi = self.mod_layers[i - 1](h)
            out = torch.sin(self.omega_0 * (phi_psi[:self.hidden_size] * self.hid_layers[i](out)) + phi_psi[self.hidden_size:])

        return  torch.sin(self.omega_0 * self.last_layer(out))