# validate_dqn.py
import os, time, argparse, csv, math, random, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Bernoulli, Normal
from spikingjelly.activation_based import functional, neuron, surrogate, monitor
from config import SimulationConfig
DEBUG_MONITOR = False
cfg = SimulationConfig()

"""
사용중인 것들: cerebellarcnnac2
"""

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

class SelfInhibitLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def reset(self):
        if isinstance(self.v, torch.Tensor):
            with torch.no_grad():
                self.v.zero_()
    def single_step_forward(self, x):
        self.v_float_to_tensor(x)
    

class CerebellarCNNAC2_old(nn.Module):
    def __init__(self, n_grc=cfg.n_grc, n_pkg=cfg.n_pkj, n_motor=4, tau=cfg.tau, T=cfg.T, std = 0.0, log = False):
        super(CerebellarCNNAC2, self).__init__()

        self.n_grc_flat = n_grc*16
        motor_decay = cfg.motor_decay

        self.critic = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = n_grc, kernel_size= 4, stride = 2, padding = 'valid', padding_mode = 'zeros', bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Flatten(),
            nn.Linear(self.n_grc_flat, 1, bias = False),
            NonSpikingLIFNode(tau = tau)
        )
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = n_grc, kernel_size= 4, stride = 2, padding = 'valid', padding_mode = 'zeros', bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Flatten(),
            nn.Linear(self.n_grc_flat, n_motor, bias = False),
            NonSpikingLIFNode(tau = tau)
        )
        self.log_std = nn.Parameter(torch.ones(1, 4) * std)
        self.log = log
        self.T = T
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init =="kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
            
            if isinstance(m, nn.Conv2d):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init =="kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data)
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

class CerebellarCNNAC2_old_2(nn.Module):
    def __init__(self, n_grc=8, n_pkj=32, n_motor=4, tau=cfg.tau, T=cfg.T, std = 0.0, log = False):
        super(CerebellarCNNAC2_old_2, self).__init__()

        self.n_grc = n_grc*16
        self.n_pkj = n_pkj*9

        self.critic = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = n_grc, kernel_size= 2, stride = 1, padding = 'valid', padding_mode = 'zeros', bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Conv2d(in_channels = n_grc, out_channels = n_pkj, kernel_size= 2, stride = 1, padding = 'valid', padding_mode = 'zeros', bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Flatten(),
            nn.Linear(n_pkj, 1, bias = False),
            NonSpikingLIFNode(tau = cfg.motor_decay)
        )
        
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = n_grc, kernel_size= 4, stride = 2, padding = 'valid', padding_mode = 'zeros', bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Conv2d(in_channels = n_grc, out_channels = n_pkj, kernel_size= 2, stride = 1, padding = 'valid', padding_mode = 'zeros', bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Flatten(),
            nn.Linear(self.n_pkj, n_motor, bias = False),
            NonSpikingLIFNode(tau = cfg.motor_decay)
        )
        self.log_std = nn.Parameter(torch.ones(1, 4) * std)
        self.log = log
        self.T = T
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init =="kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
            
            if isinstance(m, nn.Conv2d):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init =="kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data)
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
        functional.reset_net(self)
        for t in range(self.T):
            self.critic(x)
            self.actor(x)
        value = self.critic[-1].v.clone()
        mu = self.actor[-1].v.clone()
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        #if DEBUG: print(f"[TRAIN] actor:{actor}, critic:{critic}, actor potential:{self.actor[-1].v}, critic potential:{self.critic[-1].v},")
        return dist, value



class CerebellarCNN(nn.Module):
    def __init__(self, n_grc=32, n_pkj=4, n_motor=4, tau=cfg.tau, T=cfg.T, std = 0.0, log = False):
        super(CerebellarCNN, self).__init__()

        self.n_grc = n_grc*16
        self.n_pkj = n_pkj*9
        self.conv0_actor = nn.Conv2d(1, 1, kernel_size = 1, stride = 1, padding='valid', bias = False)
        self.lif0_actor = neuron.LIFNode(tau=cfg.integral_decay, surrogate_function=surrogate.ATan(), detach_reset=False)
        self.conv1_actor = nn.Conv2d(2, n_grc, kernel_size=4, stride=2, padding='valid', bias=False)
        self.lif1_actor = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=False)

        #self.conv2_actor = nn.Conv2d(self.n_grc, n_pkj, kernel_size=2, stride=1, padding='valid', bias=False)
        self.lif2_actor = neuron.LIFNode(tau=cfg.t_pkj, surrogate_function=surrogate.ATan(), detach_reset=False)

        self.flatten_actor = nn.Flatten()
        self.pkj_actor = nn.Linear(self.n_grc, self.n_pkj, bias=False)
        self.linear_actor = nn.Linear(self.n_pkj, n_motor, bias=False)
        self.lif_out_actor = NonSpikingLIFNode(tau=cfg.motor_decay)

        self.log_std = nn.Parameter(torch.ones(1, 4) * std)
        self.log = log
        self.T = T
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init =="kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
            
            if isinstance(m, nn.Conv2d):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init =="kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
            # monitoring setting
            if isinstance(m, neuron.LIFNode):
                m.store_v_seq = False
    def forward(self, x):
        dx=x.clone()
        x=self.lif0_actor(self.conv0_actor(x))
        #print(x.shape, dx.shape)
        if x.shape[0] < dx.shape[0]:
            x=x.repeat(dx.shape[0], 1, 1, 1)
        elif x.shape[0] > dx.shape[0]:
            x=x[0].unsqueeze(0)
        #print(x.shape)
        x = torch.cat((x, dx), dim = -3)
        x = self.conv1_actor(x)
        x = self.lif1_actor(x)
        x = self.flatten_actor(x)

        #x = self.conv2_actor(x)
        x = self.pkj_actor(x)
        x = self.lif2_actor(x)

        x = self.linear_actor(x)
        x = self.lif_out_actor(x)
        return x
    def set_logging(self, flag: bool):
        self.log = flag
        if self.log == True:
            self.input_monitor = monitor.InputMonitor(net = self, instance = neuron.LIFNode)
            self.spike_monitor = monitor.OutputMonitor(net=self, instance=neuron.LIFNode)
            self.potential_monitor = monitor.AttributeMonitor(net=self, pre_forward=False, instance=neuron.LIFNode, attribute_name='v')
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



class CerebellarCNNAC2(nn.Module):
    def __init__(self, n_grc=32, n_pkj=4, n_motor=4, tau=cfg.tau, T=cfg.T, std = 0.0, log = False):
        super(CerebellarCNNAC2, self).__init__()

        self.n_grc = n_grc*16
        self.n_pkj = n_pkj*9
        self.conv0_actor = nn.Conv2d(1, 1, kernel_size = 1, stride = 1, padding='valid', bias = False)
        self.lif0_actor = neuron.LIFNode(tau=cfg.integral_decay, surrogate_function=surrogate.ATan(), detach_reset=False)
        self.conv1_actor = nn.Conv2d(2, n_grc, kernel_size=4, stride=2, padding='valid', bias=False)
        self.lif1_actor = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=False)

        #self.conv2_actor = nn.Conv2d(self.n_grc, n_pkj, kernel_size=2, stride=1, padding='valid', bias=False)
        self.lif2_actor = neuron.LIFNode(tau=cfg.t_pkj, surrogate_function=surrogate.ATan(), detach_reset=False)

        self.flatten_actor = nn.Flatten()
        self.pkj_actor = nn.Linear(self.n_grc, self.n_pkj, bias=False)
        self.linear_actor = nn.Linear(self.n_pkj, n_motor, bias=False)
        self.lif_out_actor = NonSpikingLIFNode(tau=cfg.motor_decay)

        self.log_std = nn.Parameter(torch.ones(1, 4) * std)
        self.log = log
        self.T = T
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init =="kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
            
            if isinstance(m, nn.Conv2d):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init =="kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
            # monitoring setting
            if isinstance(m, neuron.LIFNode):
                m.store_v_seq = False
    def forward(self, x):
        dx=x.clone()
        x=self.lif0_actor(self.conv0_actor(x))
        #print(x.shape, dx.shape)
        if x.shape[0] < dx.shape[0]:
            x=x.repeat(dx.shape[0], 1, 1, 1)
        elif x.shape[0] > dx.shape[0]:
            x=x[0].unsqueeze(0)
        #print(x.shape)
        x = torch.cat((x, dx), dim = -3)
        x = self.conv1_actor(x)
        x = self.lif1_actor(x)
        x = self.flatten_actor(x)

        #x = self.conv2_actor(x)
        x = self.pkj_actor(x)
        x = self.lif2_actor(x)

        x = self.linear_actor(x)
        x = self.lif_out_actor(x)
        return x
    def set_logging(self, flag: bool):
        self.log = flag
        if self.log == True:
            self.input_monitor = monitor.InputMonitor(net = self, instance = neuron.LIFNode)
            self.spike_monitor = monitor.OutputMonitor(net=self, instance=neuron.LIFNode)
            self.potential_monitor = monitor.AttributeMonitor(net=self, pre_forward=False, instance=neuron.LIFNode, attribute_name='v')
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


class CerebellarCNNAC2_new(nn.Module):
    def __init__(self, n_grc=32, n_pkj=4, n_motor=4, tau=cfg.tau, T=cfg.T, std = 0.0, log = False):
        super(CerebellarCNNAC2, self).__init__()

        self.n_grc = n_grc*16
        self.n_pkj = n_pkj*9
        self.conv0_actor = nn.Conv2d(1, 1, kernel_size = 1, stride = 1, padding='valid', bias = False)
        self.lif0_actor = neuron.LIFNode(tau=cfg.integral_decay, surrogate_function=surrogate.ATan(), detach_reset=False)
        self.conv1_actor = nn.Conv2d(2, n_grc, kernel_size=4, stride=2, padding='valid', bias=False)
        self.lif1_actor = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=False)

        #self.conv2_actor = nn.Conv2d(self.n_grc, n_pkj, kernel_size=2, stride=1, padding='valid', bias=False)

        self.flatten_actor = nn.Flatten()
        self.pkj_actor = nn.Linear(self.n_grc, self.n_pkj, bias=False)
        self.lif2_actor = neuron.LIFNode(tau=cfg.t_pkj, surrogate_function=surrogate.ATan(), detach_reset=False)
        self.linear_actor = nn.Linear(self.n_pkj, n_motor, bias=False)
        self.lif_out_actor = NonSpikingLIFNode(tau=cfg.motor_decay)

        self.log_std = nn.Parameter(torch.ones(1, 4) * std)
        self.log = log
        self.T = T
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init =="kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
            
            if isinstance(m, nn.Conv2d):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init =="kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
            # monitoring setting
            if isinstance(m, neuron.LIFNode):
                m.store_v_seq = False
    def forward(self, x):
        dx=x.clone()
        x=self.lif0_actor(self.conv0_actor(x))
        #print(x.shape, dx.shape)
        # 단일 forward / eval 간의 배치 사이즈 맞추기
        if x.shape[0] < dx.shape[0]:
            x=x.repeat(dx.shape[0], 1, 1, 1)
        elif x.shape[0] > dx.shape[0]:
            x=x[0].unsqueeze(0)
        #print(x.shape)
        x = torch.cat((x, dx), dim = -3)
        x = self.conv1_actor(x)
        x = self.lif1_actor(x)
        x = self.flatten_actor(x)

        #x = self.conv2_actor(x)
        x = self.pkj_actor(x)
        x = self.lif2_actor(x)

        x = self.linear_actor(x)
        x = self.lif_out_actor(x)
        return x
    def set_logging(self, flag: bool):
        self.log = flag
        if self.log == True:
            self.input_monitor = monitor.InputMonitor(net = self, instance = neuron.LIFNode)
            self.spike_monitor = monitor.OutputMonitor(net=self, instance=neuron.LIFNode)
            self.potential_monitor = monitor.AttributeMonitor(net=self, pre_forward=False, instance=neuron.LIFNode, attribute_name='v')
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



"""
        self.conv_mf = nn.Conv2d(1, n_dmf, kernel_size = 1, stride = 1, padding='valid', bias = False)
        #self.conv_cf = nn.Conv1d(1, 1, kernel_size = 1, stride = 1, padding='valid', bias=False)
        self.mf2grc = nn.Conv2d(n_dmf, n_grc, kernel_size=3, stride=2, padding='same', bias = False) #5*5
        self.mf2goc = nn.Conv2d(n_dmf, n_goc, kernel_size=3, stride=2, padding='same', bias = False) 
        self.goc2grc = nn.Conv2d(n_goc, n_grc, kernel_size=3, stride=1, padding='same', bias = False)
        
        self.grc2pkj = nn.Conv2d(n_grc, n_pkj, kernel_size=3, stride=1, padding='valid', bias = False)
        self.grc2mli = nn.Conv2d(n_grc, n_mli, kernel_size=3, stride=1, padding='valid', bias=False)
        self.mli2pkj = nn.Conv2d(n_mli, n_pkj, kernel_size=3, stride=1, padding='same', bias=False) #linear 4
"""
class SpikingNet(nn.Module):
    def __init__(self, 
                 n_grc = 1024, # 1024 corresponds to 16*64c
                 n_goc = 256,
                 n_mli = 32, 
                 n_pkj = 8, 
                 n_motor=4, 
                 T=cfg.T, 
                 log = False):
        super(SpikingNet, self).__init__()
        self.mf2grc = nn.Linear(100, n_grc, bias=False) 
        self.mf2goc = nn.Linear(100, n_goc, bias=False)
        self.goc2grc = nn.Linear(n_goc, n_grc, bias=False)
        self.grc2pkj = nn.Linear(n_grc, n_pkj, bias=False)
        self.grc2mli = nn.Linear(n_grc, n_mli, bias=False)
        self.mli2pkj = nn.Linear(n_mli, n_pkj, bias=False)
        self.pkj2motor = nn.Linear(n_pkj, n_motor, bias=False)
        self.grc = neuron.LIFNode(tau=2.0, decay_input=False, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.goc = neuron.LIFNode(tau=32.0, decay_input=False, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.pkj = neuron.LIFNode(tau=2.0, decay_input=False, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.mli = neuron.LIFNode(tau=8.0, decay_input=False, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.motor =neuron.LIFNode(tau=2.0, decay_input=False, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.motor_state = NonSpikingLIFNode(tau=4.0)
        
        A = torch.tensor([
            [0,1,0,0],
            [1,0,0,0],
            [0,0,0,1],
            [0,0,1,0],
        ], dtype=torch.float32) # 반대방향 inhibit
        self.register_buffer("A_mask", A)

        self.log = log
        self.T = T
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, neuron.LIFNode):
                m.store_v_seq = False
        
    def _lateral_inhibit(self, x):
        y = motor
        inh = y @ self.A_mask
        y = motor - 0.5 * inh #alpha               # 억제 적용
    
    def forward(self, mf):
        """
        mf: 센서 입력
        cf: 전 단 모터뉴런 출력
        motor: 모터뉴런 전위(출력)
        """
        '''dmf=mf.clone()
        mf=self.lif0_actor(self.conv0_actor(mf))
        #print(x.shape, dx.shape)
        #input shape
        if mf.shape[0] < dmf.shape[0]:
            mf=mf.repeat(dmf.shape[0], 1, 1, 1)
        elif mf.shape[0] > dmf.shape[0]:
            mf=mf[0].unsqueeze(0)
        #print(x.shape)
        mf = torch.cat((mf, dmf), dim = -3)'''
        mf = torch.flatten(mf, start_dim=2)
        goc = self.goc(self.mf2goc(mf))
        grc = self.grc(self.mf2grc(mf)+self.goc2grc(-goc))
        mli = self.mli(self.grc2mli(grc))
        pkj = self.pkj(self.grc2pkj(grc)+self.mli2pkj(-mli))
        pkj2motor = self.pkj2motor(pkj)
        motor = self.motor(pkj2motor)
        inh = motor @ self.A_mask
        pkj2motor = pkj2motor - 0.5 * inh #alpha
        motor_state = self.motor_state(pkj2motor)
        return motor_state
        
    def set_logging(self, flag: bool):
        self.log = flag
        if self.log == True:
            self.input_monitor = monitor.InputMonitor(net=self, instance = neuron.LIFNode)
            self.spike_monitor = monitor.OutputMonitor(net=self, instance=neuron.LIFNode, )
            self.potential_monitor = monitor.AttributeMonitor(net=self, pre_forward=False, instance=neuron.LIFNode, attribute_name='v')
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
            
class SpikingCNN(nn.Module):
    def __init__(self, 
                 n_grc = 64*25, # 1024 corresponds to 16*64c
                 n_goc = 8*25,
                 n_mli = 64, 
                 n_pkj = 8*4, 
                 n_motor=4, 
                 T=cfg.T, 
                 log = False):
        super(SpikingCNN, self).__init__()
        c_grc = 64 # 10*10 -> 5*5*n_grc -> linear, output features = 64*25 = 1600
        c_goc = 8  # 10*10 -> 2*2*n_goc -> linear, output features = *25 = 
        c_pkj = 8
        
        self.mf2grc = nn.Conv2d(1, c_grc, kernel_size = 2, stride = 2, padding = 'valid', bias = False)
        self.mf2goc = nn.Conv2d(1, c_goc, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.pf2pkj = nn.Conv2d(c_grc+c_goc, c_pkj, kernel_size = 3, stride = 2, padding = 'valid', bias=False)
        
        self.pf2mli = nn.Linear(n_grc+n_goc, n_mli, bias=False)
        self.mli2pkj = nn.Linear(n_mli, n_pkj, bias=False)
        self.pkj2motor = nn.Linear(n_pkj, n_motor, bias=False)
        self.grc = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.goc = neuron.LIFNode(tau=32.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.pkj = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.mli = neuron.LIFNode(tau=8.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.motor =neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.motor_state = NonSpikingLIFNode(tau=16.0)
        
        A = torch.tensor([
            [0,1,0,0],
            [1,0,0,0],
            [0,0,0,1],
            [0,0,1,0],
        ], dtype=torch.float32) # 반대방향 inhibit
        self.register_buffer("A_mask", A)

        self.log = log
        self.T = T
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, neuron.LIFNode):
                m.store_v_seq = False
        
    
    def forward(self, mf):
        """
        mf: 센서 입력
        cf: 전 단 모터뉴런 출력
        motor: 모터뉴런 전위(출력)
        """
        '''dmf=mf.clone()
        mf=self.lif0_actor(self.conv0_actor(mf))
        #print(x.shape, dx.shape)
        #input shape
        if mf.shape[0] < dmf.shape[0]:
            mf=mf.repeat(dmf.shape[0], 1, 1, 1)
        elif mf.shape[0] > dmf.shape[0]:
            mf=mf[0].unsqueeze(0)
        #print(x.shape)
        mf = torch.cat((mf, dmf), dim = -3)'''
        
        goc = -self.goc(self.mf2goc(mf))
        grc = self.grc(self.mf2grc(mf))
        print(grc.shape, goc.shape)
        pf = torch.cat((grc, goc), dim=-3) #channel
        print(grc.shape, goc.shape, pf.shape)
        mli = -self.mli(self.pf2mli(torch.flatten(pf)))
        pkj = self.pkj(self.flatten(self.pf2pkj(pf)) + mli)
        pkj2motor = self.pkj2motor(pkj)
        motor = self.motor(pkj2motor)
        
        inh = motor @ self.A_mask
        pkj2motor = pkj2motor - 0.5 * inh #alpha
        motor_state = self.motor_state(pkj2motor)
        return motor_state
        
    def set_logging(self, flag: bool):
        self.log = flag
        if self.log == True:
            self.input_monitor = monitor.InputMonitor(net=self, instance = neuron.LIFNode)
            self.spike_monitor = monitor.OutputMonitor(net=self, instance=neuron.LIFNode, )
            self.potential_monitor = monitor.AttributeMonitor(net=self, pre_forward=False, instance=neuron.LIFNode, attribute_name='v')
        else:
            monitor.InputMonitor.disable()
            monitor.OutputMonitor.disable()
            monitor.AttributeMonitor.disable()
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