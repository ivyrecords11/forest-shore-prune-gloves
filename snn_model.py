# validate_dqn.py
import os, time, argparse, csv, math, random, collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Bernoulli, Normal
from spikingjelly.activation_based import functional, neuron, surrogate, monitor
from config import SimulationConfig

cfg = SimulationConfig()

class NonSpikingLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def reset(self): 
        # 기존: self.v = torch.tensor(0.0)  # 디바이스/그래프 끊기고 새 텐서 생성 -> 누수/성능저하
        if isinstance(self.v, torch.Tensor):
            with torch.no_grad():
                self.v.zero_()


    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)

        if self.training:
            self.neuronal_charge(x)
        else:
            if self.v_reset is None:
                if self.decay_input:
                    self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
                else:
                    self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)
                
            else:
                if self.decay_input:
                    self.v = self.neuronal_charge_decay_input(x, self.v, self.v_reset, self.tau)
                else:
                    self.v = self.neuronal_charge_no_decay_input(x, self.v, self.v_reset, self.tau)
        return self.v

class CerebellarCNNAC2(nn.Module):
    def __init__(self, n_grc=8, n_pkg=32, n_motor=4, tau=cfg.tau, T=cfg.T, std = 0.0, log = False):
        super(CerebellarCNNAC2, self).__init__()

        self.n_grc = n_grc*36
        motor_decay = cfg.motor_decay

        self.critic = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = n_grc, kernel_size= 5, stride = 1, padding = 'valid', padding_mode = 'zeros', bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Flatten(),
            nn.Linear(self.n_grc, 1, bias = False),
            NonSpikingLIFNode(tau = tau)
        )
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = n_grc, kernel_size= 5, stride = 1, padding = 'valid', padding_mode = 'zeros', bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Flatten(),
            nn.Linear(self.n_grc, n_motor, bias = False),
            NonSpikingLIFNode(tau = tau)
        )
        self.log_std = nn.Parameter(torch.ones(1, 4) * std)
        self.log = log
        self.T = T
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
            
            if isinstance(m, nn.Conv2d):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)

            # monitoring setting
            if isinstance(m, neuron.LIFNode):
                m.store_v_seq = False
        
    def set_logging(self, flag: bool):
        self.log = flag
        if self.log == True:
            self.input_monitor = monitor.InputMonitor(net=self.actor, instance = neuron.LIFNode)
            self.spike_monitor = monitor.OutputMonitor(net=self.actor, instance=neuron.LIFNode, )
            self.potential_monitor = monitor.AttributeMonitor(net=self.actor, pre_forward=False, instance=neuron.LIFNode, attribute_name='v')
        else:
            self.input_monitor = None
            self.spike_monitor = None
            self.potential_monitor = None

    def return_potential_monitor(self):
        if self.log==True:
            #if DEBUG: print(f"[TRAIN] spike monitor records: {self.spike_monitor.records}")n
            step_potential = self.potential_monitor['1'][-1].detach().squeeze().cpu().numpy()
            #step_potential = np.array(self.potential_monitor['1'].detach())
            self.potential_monitor.clear_recorded_data()
            if DEBUG_MONITOR: print(f"[TRAIN] step_potential = {step_potential.flatten()[:20]}...")
            return step_potential
        else: return False
    def return_spike_monitor(self):
        if self.log==True:
            #if DEBUG: print(f"[TRAIN] spike monitor records: {self.spike_monitor.records}")
            step_spike = self.spike_monitor['1'][-1].detach().squeeze().cpu().numpy()
            if DEBUG_MONITOR: print(type(step_spike))
            self.spike_monitor.clear_recorded_data()
            if DEBUG_MONITOR: print(f"[TRAIN] step_spike = {step_spike.flatten()[:20]}...")
            return step_spike
        else: print("error")
    def clear_monitor(self):
        if self.log == True:
            self.spike_monitor.clear_recorded_data()
            self.potential_monitor.clear_recorded_data()
            self.input_monitor.clear_recorded_data()
            
        
    def forward(self, x):
        # ... (SNN 순전파 코드)
        for t in range(self.T):
            self.critic(x)
            self.actor(x)
        value = self.critic[-1].v
        mu = self.actor[-1].v
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        #if DEBUG: print(f"[TRAIN] actor:{actor}, critic:{critic}, actor potential:{self.actor[-1].v}, critic potential:{self.critic[-1].v},")
        return dist, value

class CerebellarCNNAC2(nn.Module):
    def __init__(self, n_grc=8, n_pkg=32, n_motor=4, tau=cfg.tau, T=cfg.T, std = 0.0, log = False):
        super(CerebellarCNNAC2, self).__init__()

        self.n_grc = n_grc*36
        motor_decay = cfg.motor_decay

        self.critic = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = n_grc, kernel_size= 5, stride = 1, padding = 'valid', padding_mode = 'zeros', bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Flatten(),
            nn.Linear(self.n_grc, 1, bias = False),
            NonSpikingLIFNode(tau = tau)
        )
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = n_grc, kernel_size= 5, stride = 1, padding = 'valid', padding_mode = 'zeros', bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Flatten(),
            nn.Linear(self.n_grc, n_motor, bias = False),
            NonSpikingLIFNode(tau = tau)
        )
        self.log_std = nn.Parameter(torch.ones(1, 4) * std)
        self.log = log
        self.T = T
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
            
            if isinstance(m, nn.Conv2d):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)

            # monitoring setting
            if isinstance(m, neuron.LIFNode):
                m.store_v_seq = False
        
    def set_logging(self, flag: bool):
        self.log = flag
        if self.log == True:
            self.input_monitor = monitor.InputMonitor(net=self.actor, instance = neuron.LIFNode)
            self.spike_monitor = monitor.OutputMonitor(net=self.actor, instance=neuron.LIFNode, )
            self.potential_monitor = monitor.AttributeMonitor(net=self.actor, pre_forward=False, instance=neuron.LIFNode, attribute_name='v')
        else:
            self.input_monitor = None
            self.spike_monitor = None
            self.potential_monitor = None

    def return_potential_monitor(self):
        if self.log==True:
            #if DEBUG: print(f"[TRAIN] spike monitor records: {self.spike_monitor.records}")n
            step_potential = self.potential_monitor['1'][-1].detach().squeeze().cpu().numpy()
            #step_potential = np.array(self.potential_monitor['1'].detach())
            self.potential_monitor.clear_recorded_data()
            if DEBUG_MONITOR: print(f"[TRAIN] step_potential = {step_potential.flatten()[:20]}...")
            return step_potential
        else: return False
    def return_spike_monitor(self):
        if self.log==True:
            #if DEBUG: print(f"[TRAIN] spike monitor records: {self.spike_monitor.records}")
            step_spike = self.spike_monitor['1'][-1].detach().squeeze().cpu().numpy()
            if DEBUG_MONITOR: print(type(step_spike))
            self.spike_monitor.clear_recorded_data()
            if DEBUG_MONITOR: print(f"[TRAIN] step_spike = {step_spike.flatten()[:20]}...")
            return step_spike
        else: print("error")
    def clear_monitor(self):
        if self.log == True:
            self.spike_monitor.clear_recorded_data()
            self.potential_monitor.clear_recorded_data()
            self.input_monitor.clear_recorded_data()
            
        
    def forward(self, x):
        # ... (SNN 순전파 코드)
        for t in range(self.T):
            self.critic(x)
            self.actor(x)
        value = self.critic[-1].v
        mu = self.actor[-1].v
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        #if DEBUG: print(f"[TRAIN] actor:{actor}, critic:{critic}, actor potential:{self.actor[-1].v}, critic potential:{self.critic[-1].v},")
        return dist, value
