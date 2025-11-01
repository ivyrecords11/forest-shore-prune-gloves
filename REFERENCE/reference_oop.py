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
# initial value - must be all programmable
N_INPUT = 100
BUFFER_SIZING = 'maximum' # or optimal
OPTIMAL_BUFFER_SIZE = 0
THRESHOLD = 1

C_GRC   = cfg.c_grc;    C_GOC   = cfg.c_goc;    C_BKC   = cfg.c_bkc;    C_PKJ   = cfg.c_pkj
N_GRC   = cfg.n_grc;    N_GOC   = cfg.n_goc;    N_BKC   = cfg.n_bkc;    N_PKJ   = cfg.n_pkj;    N_MOTOR = 4
TAU_GRC = cfg.t_grc;    TAU_GOC = cfg.t_goc;    TAU_BKC = cfg.t_bkc;    TAU_PKJ = cfg.t_pkj;    TAU_MOTOR = cfg.motor_decay
F_GRC   = cfg.f_grc;    F_GOC   = cfg.f_goc;    F_G2G   = cfg.f_g2g;    F_BKC   = cfg.f_bkc;    F_PKJ   = cfg.f_pkj;    F_B2P   = cfg.f_b2p

# MEMORY-WEIGHTS( cpre*cpost*filter_w**2)
# load from file later
w_grc = np.zeros(    1*C_GRC*F_GRC*F_GRC)
w_goc = np.zeros(    1*C_GOC*F_GOC*F_GOC)
w_g2g = np.zeros(C_GOC*C_GRC*F_GOC*F_GOC)
w_pkj = np.zeros(C_GRC*C_PKJ*F_PKJ*F_PKJ)
w_bkc = np.zeros(C_GRC*C_BKC*F_B2P*F_B2P)
w_b2p = np.zeros(C_BKC*C_PKJ*F_B2P*F_B2P)
w_p2m = np.zeros(N_PKJ*    1*    2*    2)
# this is also the address obtaining formula (pre channel * post channel * height * width)

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
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size)
        self.ptr: int
    # deleted reset due to optimizing reasons
    '''def reset(self):
        self.buffer[:] = 0 # clear
        self.ptr: int = 0'''
    def add_buf(self, out_addr):
        self.buffer[self.ptr] = out_addr
    def done_buf(self):
        self.buffer[self.ptr] = -9999 #exit code: 2**self.max_addr_width - 1 #1111_..._1111
        self.ptr = 0
    def loc_buf(self): # valid 없어도 되는거같은데
        out = self.buffer[self.ptr]
        self.buffer[self.ptr] = 0
        self.ptr += 1
        if self.buffer == self.ptr: 
            self.reset()
        return out

LAYERS = {'input': 0b000, 'goc': 0b001, 'grc': 0b011, 'bkc':0b100, 'pkj':0b101, 'motor':0b111} #0,1,2,3
# last bit goes to mux.
print(LAYERS)

'''
input_buffer = BinaryBuffer(LAYERS['input'], N_INPUT)
grc_buffer = BinaryBuffer(LAYERS['grc'], N_GRC)
goc_buffer = BinaryBuffer(LAYERS['goc'], N_GOC)
bkc_buffer = BinaryBuffer(LAYERS['bkc'], N_BKC)
pkj_buffer = BinaryBuffer(LAYERS['pkj'], N_PKJ)
motor_buffer = BinaryBuffer(LAYERS['motor'], N_MOTOR)
buffer_array = [input_buffer, grc_buffer, goc_buffer, bkc_buffer, pkj_buffer, motor_buffer]'''
# choose buffer sizing method
# 계산 순서
# 1. mf->grc 1회 돌림 (mf, grc 버퍼 존재)
# 2. mf->goc 1회 돌림 (mf, goc)
# 3. goc->grc

if BUFFER_SIZING == 'maximum':
    BUFFER_0_SIZE = max(N_INPUT, N_GRC, N_GOC, N_BKC)
    BUFFER_1_SIZE = max(N_GRC, N_GOC, N_BKC, N_PKJ)
    BUFFER_2_SIZE = max(N_GRC, n_)
BUFFER_ADDR = math.ceil(math.log2(INPUT_BUFFER_SIZE))
spike_buffer = BinaryBuffer(INPUT_BUFFER_SIZE)

def input_transform(inputs: np.array):
    for i in range(N_INPUT): # i => counter
        if input[i]:
            spike_buffer.add_buf(i)
    spike_buffer.done_buf() # command
    
class WeightStationaryConvolution:
    def __init__(self, image_width: int, in_channel_width: int, out_channel_width: int, window_width: int, stride: int, tau: int):
        #padding is 0 in all cases
        self.image_width = image_width
        self.flattened_width = in_channel_width * out_channel_width * window_width**2
        self.stride = stride
        self.tau = tau
        self.acc: int
        self.out_ptr: int
        
    def reset(self):
        self.acc = 0
        self.out_ptr = 0
    
    def _find_output_pixel:
        
        
    def add(self, addr):
        #depending on the output
        while True:
            addr = self.buffer.loc_buf()
            #TODO: convert addr into weight address using window width and stride
            
            #logic to see if it is in range
            if addr == -9999:
                break
        if acc >= THRESHOLD:
            
            acc -= THRESHOLD

class OutputStationaryConvolution:
    def __init__(self):
        #padding is 0 in all cases
        self.image_width = image_width
        self.flattened_width = in_channels * out_channels * window_width**2
        self.stride = stride
        self.tau = tau
        self.acc: int
        self.out_ptr: int
    
    def set_layer(self, image_width: int, in_channels: int, out_channels: int, window_width: int, stride: int, tau: int):
        self.image_width = image_width
        self.window_width = window_width
        self.flattened_width = in_channels * out_channels * window_width**2
        self.stride = stride
        self.tau = tau
        
    def reset(self):
        self.acc = 0
        self.out_ptr = 0
        
    def add(self, addr):
        #depending on the output
        while True: # iter for all inputs in queue
            addr = self.buffer.loc_buf() # input address
            for r in range(window_width):
                for c in range(window_width)
            #TODO: convert addr into weight address using window width and stride
            # if not in accumulator range: cut
            # if longer than acc range: cut -> how.
            #logic to see if it is in range
            
            
            if addr == -9999:
                break
        if acc >= THRESHOLD:
            
            acc -= THRESHOLD
        
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