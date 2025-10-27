from mujoco_model_v10 import Environment
from utils_logger import SpikePlotter, SpikeHeatmap

import numpy as np

@dataclass
class CerebellumConfig:
    mf = {"n" : 100}
    grc = {
        "n": int = 512, 
        "tau": float = 0.933
        }
    goc = 

class LIFNeuron:
    """
    fixed: tau
    Members: v, thr, 
    methods: get_potential, reset (to 0)
    """
    def __init__(self, tau, threshold_init = 1, self_inhibit = False, th_increase=1.0, DEBUG = False):
        #initialize variables
        """
        tau
        tau_th
        threshold_init
        th_increase_ratio
        """
        self._tau = tau
        self._th_increase = th_increase
        self.v = 0
        self.threshold_init = threshold_init
        self.threshold = threshold_init
        self.self_inhibit = self_inhibit
        self.spike = False
        
    def step(self, z: np.array):
        """
        input: list of inputs.
        does 3 things: adds v. fires/adjusts threshold. lowers threshold."""
        self.v *= (self._tau-1)/(self._tau) #decay
        self.v += z
        
        if self.v>self.threshold:
            self.spike = True
            self.v=0
            
        if self.self_inhibit:
            if self.spike:
                self.threshold += self.threshold * self._th_increase
            self.threshold -= (self.threshold-self.threshold_init)*self.tau-1
            
        if DEBUG: print(f"step: v {self.v}\t threshold {self.threshold}\t")
        return self.spike
    def get_potential(self):
        return self.v
    #def inhibit_self(self, threshold):
    def reset(self):
        self.v = 0
        self.spike = False
        self.threshold = self.threshold_init
        
class Linear:
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
    def forward(self, x: np.array, weight: np.array, output: np.array):
        assert x.shape==(1,self.inputs), f"shape mismatch: {x.shape}!={(1,self.inputs)}"
        assert weight.shape[0]==(self.inputs, self.outputs), f"shape mismatch: {x.shape[0]}!={self.inputs}"
        
class Cerebellum:
    def __init__(self, n_grc, n_goc, n_pkj, n_cf):
        """cf: and """
        self.n_grc = n_
    
if __name__=="__main__":
    
    #initialize modules