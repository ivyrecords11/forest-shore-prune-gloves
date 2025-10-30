from config import SimulationConfig
from mujoco_model_v3 import Environment
from utils_logger import SpikePlotter, SpikeHeatmap

import numpy as np
cfg = SimulationConfig

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

N_GRC   = 1024
N_GOC   = 256
N_BKC   = 32
N_PKJ   = 8
N_MOTOR = 4

TAU_GRC = 2
TAU_GOC = 32
TAU_BKC = 2
TAU_PKJ = 8
TAU_MOTOR = 2

v_grc = np.zeros(N_GRC)
v_goc = np.zeros(N_GOC)
v_pkj = np.zeros(N_PKJ)
v_bkc = np.zeros(N_BKC)
v_motor = np.zeros(N_MOTOR)
v_motor_output = np.zeros(N_MOTOR)

s_grc = np.zeros(N_GRC)
s_goc = np.zeros(N_GOC)
s_pkj = np.zeros(N_PKJ)
s_bkc = np.zeros(N_BKC)
s_motor = np.zeros(N_MOTOR)

action = np.array([0,0,0,0])
sim_dur_step = int(cfg.simulation_duration_s/cfg.dt)

for i in range(sim_dur_step):
    # load inputs from database
    # load weights from database
    # initialize neuron voltage
    """
    (1*c)(c*r)
    input image (channel, height, width)
    """
    mf, terminated, truncated, info = env.step
    
    for h in range(5):
        for w in range(5):
            print(:2d, end=" ")
        
    
    
    
    
    
    
    
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
        
    '''for simulation_step in range(sim_dur_step):
        # Conv - stride 2
        for c in range(4):
            for r in range(4):
                for i in range(4):
                    for j in range(4):'''
                    
