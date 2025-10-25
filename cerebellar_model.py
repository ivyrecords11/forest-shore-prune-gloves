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
from torch.distributions import Bernoulli, Normal
from spikingjelly.activation_based import functional, neuron, surrogate, monitor
from datetime import datetime

class WeightClipperExhibition(object):
    def __call__(self, module, param, clip_min, clip_max):
    	if hasattr(module, param):
	        self.clip(module, param, clip_min, clip_max)
        
    def clip(self, module, param, clip_min, clip_max):
        p = getattr(module, param).data
        p = p.clamp(clip_max, clip_max)
        getattr(module, param).data = p

class WeightClipperInhibition(object):
    def __call__(self, module, param, clip_min, clip_max):
    	if hasattr(module, param):
	        self.clip(module, param, clip_min, clip_max)
        
    def clip(self, module, param, clip_min, clip_max):
        p = getattr(module, param).data
        p = p.clamp(clip_max, clip_max)
        getattr(module, param).data = p

class Cerebellum(nn.Module):
    def __init__(self, cfg: SimulationConfig):
        super.__init__()
        self.cfg = cfg

        self.grc = neuron.LIFnode(tau=cfg.tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.goc = neuron.LIFnode(tau=cfg.tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.pkj = neuron.LIFnode(tau=cfg.tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.mli = neuron.LIFnode(tau=cfg.tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        

        self.e_mf2grc = nn.Conv2d(1, cfg.n_grc, 5, 1, "valid", bias=False)
        self.e_mf2goc = nn.Conv2d(1, cfg.n_goc, 3, 1, "valid", bias=False)
        self.i_goc2grc = nn.Conv2d(cfg.n_goc, cfg.n_grc, 3, 1, "valid", bias=False)
    def forward(mf, cf, ):
        x = 
    def clip_exhibition():
    def clip_inhibition():
1