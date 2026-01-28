#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Siren network model for Implicit Neural Representation. Inspired by https://github.com/MIAGroupUT/IDIR

"""

import numpy as np
import torch
from torch import nn


class Siren(nn.Module):
    """This is a dense neural network with sine activation functions.

    Arguments:
    layers -- ([*int]) amount of nodes in each layer of the network, e.g. [2, 16, 16, 1]
    weight_init -- (boolean) use special weight initialization if True
    omega -- (float) parameter used in the forward function
    init_factor -- (float) parameter used in the initialization / alternative to default SIREN
    """

    def __init__(self, layers, weight_init=True, omega=30, init_factor = None):
        """Initialize the network."""

        super(Siren, self).__init__()
        self.n_layers = len(layers) - 1
        self.omega0 = omega

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

            # Weight Initialization
            if weight_init:
                with torch.no_grad():
                    if i == 0:
                        self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
                    else:
                        if init_factor is None:
                            self.layers[-1].weight.uniform_(
                               -np.sqrt(6 /   layers[i]) / self.omega0,
                                np.sqrt(6 /  layers[i]) / self.omega0,
                               #- init_factor, init_factor
                            )
                        else:
                            self.layers[-1].weight.uniform_(
                                - init_factor, init_factor
                            )

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward function of the network."""

        # Perform sine on all layers except for the last one
        for layer in self.layers[:-1]:
            x = torch.sin(self.omega0 * layer(x))

        return self.layers[-1](x)