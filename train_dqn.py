from config import SimulationConfig
from mujoco_model_v3 import Environment
from utils_logger import SpikePlotter, SpikeHeatmap, PotentialPlotter, PotentialPlotter_vertical
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

import numpy as np
import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
from spikingjelly.activation_based import functional, neuron, surrogate, monitor
from datetime import datetime

DEBUG = True
cfg = SimulationConfig()

class NonSpikingLIFNode(neuron.LIFnode):
    def forward(self, dv: torch.Tensor):
        self.neuronal_charge(dv)
        return self.v

class CerebellarCNN(nn.Module):
    def __init__(self, cfg: SimulationConfig):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = cfg.n_grc, kernel_size= 5, stride = 1, padding = 'valid', padding_mode = 'zeros', bias = False),
            neuron.LIFNode(tau=cfg.tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Flatten(),
            nn.Linear(in_features = self.n_grc, out_features = cfg.n_motor, bias = False),
            #NonSpikingLIFNode(tau = tau)
        )
    def forward(self, x):
        for t in range(cfg.T):
            self.fc(x)
        
        return self.fc[-1].v
