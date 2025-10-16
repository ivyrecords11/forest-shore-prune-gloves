from config import SimulationConfig
from mujoco_model_v3 import Environment
from utils_logger import SpikePlotter, SpikeHeatmap

import numpy as np

DEBUG = 

"""
Conv
Fc
Fc
"""

# TODO: Parameter 불러오기
# TODO: 뉴런 설계
# TODO: 연산기 설계
# TODO: 각 단계 컨트롤러 설계 - 안해도 될듯?
# TODO: 각 단계 save


cfg = SimulationConfig(
 # SIMULATION PARAMETERS
    dt                      = 1e-3,                            # 시뮬레이션 시간 간격 (ms) 1
    eta                     = 1e-3,                           # 학습률 (alpha)
    seed                    = None,                            # 난수 시드
    T                       = 16,                                 # n개의 행동을 하고 확률분포를 구함.

    # SENSOR PARAMETERS
    n_sensor_1d             = 10,
    n_mf                    = 100,                            # 센서 개수     
    max_firing_rate         = 1000,               # 센서 최대 발화율 (Hz)
    transmitt_speed         = 50,                 # 신경 전송 속도 [m/s]
    sigma                   = 0.02,

    # NEURONAL PARAMETERS
    tau_pre_ms              = 5.0,                     # STDP 전-시냅스 가중치 변화 시간상수 (ms)
    tau_post_ms             = 5.0,                    # STDP 후-시냅스 가중치 변화 시간상수 (ms)
    tau_e_ms                = 50,                        # 시냅스 효율 시간상수 (s)
    tau                     = 2.0,

    n_grc                   = 64,
    n_pkj                   = 8,
    n_motor = 4,      # 제어할 모터 수
    pkj_threshold           = 4.0,

    # TRAINING PARAMETERS
    # every reward scale is per second
    simulation_duration_s   = 10.0,         # 각 에피소드 시뮬레이션 시간 (초)
    success_duration_s      = 3.0,             # 성공으로 간주할 연속 시간 (초)
    num_steps               = 64,                          # rollout length before an update (A2C)
    success_reward_per_sec  = 10000,
    penalize_failure_per_sec = 10000,

    # PHYSICAL PARAMETERS
    plate_size = 0.3, # (m)
    ball_mass = None,                     # 공 질량 (kg), None이면 리셋 시 무작위
    ball_density = 2000,                    # kg/m^3

    # MOTOR PARAMETERS
    # unused
    motor_max_hinge_deg = 10.0,                  # 판 최대 경사각 [deg]  
    motor_window_len = 0.01,                     # [s] 모터가 반응을 합산할 시간창(예: 20ms)
    use_ema = False,                       # True면 EMA, False면 슬라이딩 평균
    ema_alpha = 0.2,       
)


"""
Model Structire

5x5 Conv (10*10 -> 6*6)
flatten (64*6*6 -> 2048)
FC (2048 -> 32)
FC (32 -> 4)
"""


def load_weights(filepath: str):
    #TODO
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

def convolution(newrow, newcolumn)
    
def simulation(cfg: SimulationConfig):
    sim_dur_step = int(cfg.simulation_duration_s/cfg.dt)
    # load inputs from database
    # load weights from database

    # initialize neuron voltage

    v_grc = np.zeros(cfg.n_grc, 6, 6)
    threshold_grc = np.ones(cfg.n_grc) * 1.0
    v_pkj = np.zeros(cfg.n_pkj)
    threshold_pkj = np.ones(cfg.n_pkj) * cfg.pkj_threshold
    v_motor = np.zeros(cfg.n_motor)
    threshold_motor = np.ones(cfg.n_motor) * 1.0

    for simulation_step in range(sim_dur_step):
        # Conv
        for c in range(6):
            for r in range(6):
                for i in range(5):
                    for j in range(5):
        # TODO: grc -> pkj fc
        # TODO: pkj -> motor fc
        
        # TODO: 
