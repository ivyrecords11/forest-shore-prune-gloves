import os, time, argparse, csv, math, random, collections, gc
import sys

# 현재 파일 기준으로 상위 폴더 경로 추가
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from dqn_5.config_cnn2 import SimulationConfig
from mujoco_model_v11 import Environment
from utils_logger import SpikePlotter, SpikeHeatmap

import numpy as np
cfg = SimulationConfig()

DEBUG = True

# TODO: Parameter 불러오기
# TODO: 뉴런 설계
# TODO: 연산기 설계
# TODO: 각 단계 컨트롤러 설계 - 안해도 될듯?
# TODO: 각 단계 save

def quantize_weights(filepath):
    return False

def load_weights_from_manifest(filepath: str):
    #TODO
    # cfg = SimulationConfig(params)
    return 0
def save_outputs_to_file(filepath: str):
    #TODO
    #open directory as f
    return 0
class wip_LIFNeuron():
    
    """
    input: v(t), I(t)
    output: v(t+1), spike(t)
    """
    def __init__(v, I, v_th, reset = 0.0):
        v_reset = 0.0
        dt = cfg.dt
        tau = cfg.tau
        dv = ( -v + I ) * (dt / tau)
        v = v + dv
        if v >= v_th:
            spike = 1
            v = v_reset
        else:
            spike = 0
        return v, spike


"""
params/initialization
1. All thresholds are 1.
"""
env = Environment(cfg)
# initial value.
N_INPUT = 100
C_GRC   = cfg.c_grc;    C_GOC   = cfg.c_goc;    C_BKC   = cfg.c_bkc;    C_PKJ   = cfg.c_pkj
N_GRC   = cfg.n_grc;    N_GOC   = cfg.n_goc;    N_BKC   = cfg.n_bkc;    N_PKJ   = cfg.n_pkj;    N_MOTOR = 4
TAU_GRC = cfg.t_grc;    TAU_GOC = cfg.t_goc;    TAU_BKC = cfg.t_bkc;    TAU_PKJ = cfg.t_pkj;    TAU_MOTOR = cfg.motor_decay
F_GRC   = cfg.f_grc;    F_GOC   = cfg.f_goc;    F_G2G   = cfg.f_g2g;    F_BKC   = cfg.f_bkc;    F_PKJ   = cfg.f_pkj;    F_B2P   = cfg.f_b2p

# MEMORY - POTENTIAL
v_grc = np.zeros(N_GRC)
v_goc = np.zeros(N_GOC)
v_pkj = np.zeros(N_PKJ)
v_bkc = np.zeros(N_BKC)
v_motor = np.zeros(N_MOTOR)
v_motor_output = np.zeros(N_MOTOR)

# SPIKES - dunno i will use this
s_grc = np.zeros(N_GRC)
s_goc = np.zeros(N_GOC)
s_pkj = np.zeros(N_PKJ)
s_bkc = np.zeros(N_BKC)
s_motor = np.zeros(N_MOTOR)


action = np.array([0,0,0,0])
sim_dur_step = int(cfg.simulation_duration_s/cfg.dt)

class BinaryBuffer:
    def __init__(self, buffer_id, size):
        self.buffer_id = buffer_id
        self.buffer = np.zeros(size)
        self.ptr: int = 0
        self.max_addr_width = math.ceil(math.log2(size))
    def reset(self):
        self.buffer[:] = 0
        self.ptr: int = 0
        self.max_addr_width = math.ceil(math.log2(size))
    def add_buffer(self, out_addr):
        self.buffer[self.ptr] = out_addr
    def done_buffer(self):
        self.buffer[self.ptr] = -9999 #2**self.max_addr_width - 1 #1111_..._1111
    def loc_buffer(self): # valid 없어도 되는거같은데
        out = self.buffer[self.ptr]
        self.ptr += 1
        if self.buffer == self.ptr: 
            self.reset()
        return out

LAYERS = {'input': 0, 'grc': 1, 'goc': 2, 'bkc':3, 'pkj':4, 'motor':5} #0,1,2,3
print(enumerate(LAYERS))

input_buffer = BinaryBuffer(LAYERS['input'], N_INPUT)
grc_buffer = BinaryBuffer(LAYERS['grc'], N_GRC)
goc_buffer = BinaryBuffer(LAYERS['goc'], N_GOC)
bkc_buffer = BinaryBuffer(LAYERS['bkc'], N_BKC)
pkj_buffer = BinaryBuffer(LAYERS['pkj'], N_PKJ)
motor_buffer = BinaryBuffer(LAYERS['motor'], N_MOTOR)

class OutputStationaryConvolution:
    def __init__(self, window_width: int, stride: int, tau: int):
        #padding is 0 in all cases
    def add(self, addr):
        #TODO: depending on the output
        #TODO: convert addr into weight address using window width and stride
class WeightStationaryConvolution:
    def __init__(self, window_width: int, stride: int, tau: int):
        #padding is 0 in all cases
    def add(self, addr):
        #TODO: depending on the output
        #TODO: convert addr into output address using window width and stride
'''

for i in range(sim_dur_step):
    # load inputs from database
    # load weights from database
    # initialize neuron voltage
    """
    (1*c)(c*r)
    input image (channel, height, width)
    """
    mf, reward, terminated, truncated, info = env.step
    
    # enqueue mf
    # [6:0]queue_mf[7]
    spike_queue_mf = [];    queue_ptr_mf == 0
    
    
    mf.reshape(100)
    for spike in mf:
        if spike:
            
            queue_ptr_mf +=1
    #mf2grc
    for h in range(5):
        for w in range(5):
            if DEBUG: print(f"{5*h+w:2d}", end=" ")
            
        print()
    
    # FC
    for r in range():
        for c in range(c):
            v_grc[r] += mf[c] * w_mf[c][r]
        if v_grc[r] >= 1:
            v_grc[r] -= 1
            s_grc[r] = 1
        v_grc -= v_grc/TAU_GRC
        

        
    

    
                        
        # TODO: grc -> pkj fc
        # TODO: pkj -> motor fc
        
        # TODO: 
        
    for simulation_step in range(sim_dur_step):
        # Conv - stride 2
        for c in range(4):
            for r in range(4):
                for i in range(4):
                    for j in range(4):
                 '''   
print(f"C_GRC: {C_GRC}\tC_GOC: {C_GOC}\tC_BKC: {C_BKC}\tC_PKJ: {C_PKJ}")
print(f"N_GRC: {N_GRC}\tN_GOC: {N_GOC}\tN_BKC: {N_BKC}\tN_PKJ: {N_PKJ}\tN_MOTOR: {N_MOTOR}")
print(f"TAU_GRC: {TAU_GRC}\tTAU_GOC: {TAU_GOC}\tTAU_BKC: {TAU_BKC}\tTAU_PKJ: {TAU_PKJ}\tTAU_MOTOR: {TAU_MOTOR}")