# validate_dqn.py
import os, time, argparse, csv, math, random, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Bernoulli, Normal
from torchinfo import summary
from spikingjelly.activation_based import functional, neuron, surrogate, monitor
from config_cnn2 import SimulationConfig
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
class NonSpikingParametricLIFNode(neuron.ParametricLIFNode):
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
    

class LIFNodeLFSR(neuron.LIFNode):
    """
    LIFNode + deterministic LFSR-based spontaneous current.
    - LFSR outputs 0 or 1 (max = 1)
    - mapped to [0, max_spont] (default 0.125)
    - updates happen under torch.no_grad() so autograd won't record them
    - LFSR state is a buffer, NOT a parameter
    """
    def __init__(self,
                 tau: float = 2.,
                 decay_input: bool = True,
                 v_threshold: float = 1.,
                 v_reset: float = 0.,
                 surrogate_function = surrogate.Sigmoid(),
                 detach_reset: bool = False,
                 step_mode='s',
                 backend='torch',
                 store_v_seq: bool = False,
                 lfsr_seed: int = 0xACE1,
                 max_spont: float = cfg.max_spont):
        super().__init__(tau=tau,
                         decay_input=decay_input,
                         v_threshold=v_threshold,
                         v_reset=v_reset,
                         surrogate_function=surrogate_function,
                         detach_reset=detach_reset,
                         step_mode=step_mode,
                         backend=backend,
                         store_v_seq=store_v_seq)
        if lfsr_seed == 0 or lfsr_seed >= (1 << 16):
            raise ValueError("lfsr_seed must be in 1..0xFFFF")
        # buffer => no grad, but saved/moved with module
        self.register_buffer("_lfsr_state", torch.tensor(lfsr_seed, dtype=torch.int32))
        self.max_spont = float(max_spont)

    @torch.no_grad()
    def _lfsr_step_scalar(self) -> int:
        """One 16-bit LFSR step, taps = 16,14,13,11 -> returns new state & output bit.
        Output bit = MSB (after shift).
        """
        s = int(self._lfsr_state.item())
        b16 = (s >> 15) & 1
        b14 = (s >> 13) & 1
        b13 = (s >> 12) & 1
        b11 = (s >> 10) & 1
        fb = b16 ^ b14 ^ b13 ^ b11
        s = ((s << 1) & 0xFFFF) | fb
        # store back
        self._lfsr_state.copy_(torch.tensor(s, dtype=torch.int32, device=self._lfsr_state.device))
        # output bit = s >> 15
        return (s >> 15) & 1

    @torch.no_grad()
    def _spontaneous_current_like(self, x: torch.Tensor) -> torch.Tensor:
        """Return tensor same shape as x, values in {0, max_spont} without grad."""
        # simplest: generate elementwise by stepping LFSR many times
        # this is python-loop but safe and deterministic
        numel = x.numel()
        bits = []
        for _ in range(numel):
            bits.append(self._lfsr_step_scalar())
        t = torch.tensor(bits, dtype=x.dtype, device=x.device).view_as(x)
        return t * self.max_spont

    def neuronal_charge(self, x: torch.Tensor):
        # make a detached spontaneous current and add to x
        with torch.no_grad():
            spont = self._spontaneous_current_like(x)
        # x may be requiring grad; we add a non-grad tensor: autograd treats it as constant
        x = x + spont
        # then run normal LIF charge using parent logic
        return super().neuronal_charge(x)


class SpikingCNN(nn.Module):
    def __init__(self, 
                 T=cfg.T, 
                 log = False):
        super(SpikingCNN, self).__init__()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            if isinstance(m, neuron.LIFNode):
                m.store_v_seq = False
        c_grc = 64
        c_goc = 8
        c_pkj = 4
        c_bkc = 16
        c_pkj = 8
        n_pkj = c_pkj*4
        n_motor = 4
        
        self.mf2goc     = nn.Conv2d(1,     c_goc, kernel_size = 5, stride = 1, padding = 0, bias=False)
        self.goc2grc    = nn.Conv2d(c_goc, c_grc, kernel_size = 2, stride = 1, padding = 0, bias=False) 
        self.mf2grc     = nn.Conv2d(1,     c_grc, kernel_size = 2, stride = 2, padding = 0, bias=False)
        self.pf2bkc     = nn.Conv2d(c_grc, c_bkc, kernel_size = 3, stride = 1, padding = 0, bias=False) #3,3
        self.bkc2pkj    = nn.Conv2d(c_bkc, c_pkj, kernel_size = 2, stride = 1, padding = 0, bias=False) #depthwise conv
        self.pf2pkj     = nn.Conv2d(c_grc, c_pkj, kernel_size = 3, stride = 2, padding = 0, bias=False) #2,2
        self.pkj2motor  = nn.Linear(n_pkj, n_motor, bias=False)
        #self.cf2pkj     = nn.Linear()
        
        self.grc = LIFNodeLFSR(tau=8.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.goc = LIFNodeLFSR(tau=32.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.pkj = LIFNodeLFSR(tau=8.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.bkc = LIFNodeLFSR(tau=16.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.motor =neuron.LIFNode(tau=cfg.motor_decay, surrogate_function=surrogate.ATan(), detach_reset=True)
        
        A = torch.tensor([
            [0,1,.1,.1],
            [1,0,.1,.1],
            [.1,.1,0,1],
            [.1,.1,1,0],
        ], dtype=torch.float32) # 반대방향 inhibit
        self.register_buffer("A_mask", A)
        self.log = log
        self.T = T #dummy
        
    
    def forward(self, mf):
        goc = -self.goc(self.mf2goc(mf))
        grc = self.grc(self.mf2grc(mf) + self.goc2grc(goc))
        bkc = -self.bkc(self.pf2bkc(grc))
        pkj = self.pkj(self.pf2pkj(grc) + self.bkc2pkj(bkc))
        pkj2motor = self.pkj2motor(torch.flatten(pkj, start_dim=1))
        motor = self.motor(pkj2motor)
        #Lateral Inhibition
        #inh = motor @ self.A_mask
        output = motor + self.motor.v
        #self.motor.v -= cfg.inhibit_rate * inh #alpha
        
        return output
    

class SpikingCNNsmall(nn.Module):
    def __init__(self, 
                 T=cfg.T, 
                 log = False):
        super(SpikingCNNsmall, self).__init__()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            if isinstance(m, neuron.LIFNode):
                m.store_v_seq = False
        c_grc = 16
        c_goc = 2
        c_bkc = 1
        c_pkj = 2
        n_pkj = c_pkj*4
        n_motor = 4
        
        self.mf2goc     = nn.Conv2d(1,     c_goc, kernel_size = 5, stride = 1, padding = 0, bias=False)
        self.goc2grc    = nn.Conv2d(c_goc, c_grc, kernel_size = 2, stride = 1, padding = 0, bias=False) 
        self.mf2grc     = nn.Conv2d(1,     c_grc, kernel_size = 2, stride = 2, padding = 0, bias=False)
        self.pf2bkc     = nn.Conv2d(c_grc, c_bkc, kernel_size = 3, stride = 1, padding = 0, bias=False) #3,3
        self.bkc2pkj    = nn.Conv2d(c_bkc, c_pkj, kernel_size = 2, stride = 1, padding = 0, bias=False) #depthwise conv
        self.pf2pkj     = nn.Conv2d(c_grc, c_pkj, kernel_size = 3, stride = 2, padding = 0, bias=False) #2,2
        self.pkj2motor  = nn.Linear(n_pkj, n_motor, bias=False)
        #self.cf2pkj     = nn.Linear()
        
        self.grc = LIFNodeLFSR(tau=8.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.goc = LIFNodeLFSR(tau=32.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.pkj = LIFNodeLFSR(tau=8.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.bkc = LIFNodeLFSR(tau=16.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.motor =neuron.LIFNode(tau=cfg.motor_decay, surrogate_function=surrogate.ATan(), detach_reset=True)
        
        A = torch.tensor([
            [0,1,.25,.25],
            [1,0,.1,.1],
            [.25,.25,0,1],
            [.25,.25,1,0],
        ], dtype=torch.float32) # 반대방향 inhibit
        self.register_buffer("A_mask", A)
        self.log = log
        self.T = T #dummy
        
    
    def forward(self, mf):
        goc = -self.goc(self.mf2goc(mf))
        grc = self.grc(self.mf2grc(mf) + self.goc2grc(goc))
        bkc = -self.bkc(self.pf2bkc(grc))
        pkj = self.pkj(self.pf2pkj(grc) + self.bkc2pkj(bkc))
        pkj2motor = self.pkj2motor(torch.flatten(pkj, start_dim=1))
        motor = self.motor(pkj2motor)
        #Lateral Inhibition
        inh = motor @ self.A_mask
        output = motor + self.motor.v
        self.motor.v -= cfg.inhibit_rate * inh #alpha
        
        return output
    

if __name__ == "__main__":
    scnn = SpikingCNNsmall()
    input_shape = (1,1,10,10)
    
    print(scnn)
    summary(scnn, input_shape)
    param_names = [name for name in scnn.state_dict().keys()]
    for pn in param_names:
        print(pn, scnn.state_dict()[pn].shape)
    input_tensor = torch.zeros((1, 1, 10, 10), dtype=torch.float)

    scnn.forward(input_tensor)
    print(scnn.grc.v)
    