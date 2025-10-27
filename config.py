from dataclasses import dataclass

@dataclass
class SimulationConfig:
    # SIMULATION PARAMETERS
    dt: float = 1e-3                            # 시뮬레이션 시간 간격 (ms) 1
    eta: float = 1e-3                           # 학습률 (alpha)
    seed: int = None                            # 난수 시드
    T: int = 1                               

    # SENSOR PARAMETERS
    n_sensor_1d: int = 10
    n_mf : int = 100                            # 센서 개수     
    max_firing_rate: float = 200                # 센서 최대 발화율 (Hz)
    transmitt_speed: float = 100                 # 신경 전송 속도 [m/s]
    sigma: float = 5                         # 센서 노이즈 표준편차 - 반지름과 곱해짐

    # NEURONAL PARAMETERS
    tau_pre_ms: float = 5.0                     # STDP 전-시냅스 가중치 변화 시간상수 (ms)
    tau_post_ms: float = 5.0                    # STDP 후-시냅스 가중치 변화 시간상수 (ms)
    tau_e_ms: float = 50                        # 시냅스 효율 시간상수 (s)
    tau: float = 4.0
    t_pkj: float = 2.0

    n_grc: int = 8 # num_filters, *36
    n_goc: int = 2
    n_pkj: int = 1
    n_mli: int = 2
    n_cf: int = 4

    t_grc: float = 2
    t_goc: float = 2
    t_mli: float = 2
    t_motor: float = 2
    th_pkj: float = 1.0
    # MODEL_PARAMETERS


    # TRAINING PARAMETERS

    weight_init: str = "kaiming"
    # every reward scale is per second
    simulation_duration_s: float = 10.0         # 각 에피소드 시뮬레이션 시간 (초)
    success_duration_s: float = 3.0             # 성공으로 간주할 연속 시간 (초)
    num_steps: int = 32                          # rollout length before an update (A2C)
    success_reward_per_sec: float = 10000
    penalize_failure_per_sec: float = 10000

    # PHYSICAL PARAMETERS
    plate_size: float = 0.3 # (m)
    ball_mass: float = None                     # 공 질량 (kg), None이면 리셋 시 무작위
    ball_density: int = 1600                    # kg/m^3, fe

    # MOTOR PARAMETERS
    n_motor: int = 4      # 제어할 모터 수
    motor_mode: str = 'spikes'                #motor_neuron or spikes
    motor_decay: float = 5.0                        #모터뉴런 복구 계수
    motor_damping: float = 3.0                      #링버퍼 제어 x - 제어 신호 - 모터 사이 순간 기울기 완화
    motor_gain: float = 1.0                        # 모터 제어 신호 이득
    motor_max_hinge_deg: float = 10.0                  # 판 최대 경사각 [deg]  
    motor_window_time: float = 0.02
    motor_window_len: int = 10                    # [s] 모터가 반응을 합산할 시간창(예: 20ms)
    use_ema: bool = False                       # True면 EMA, False면 슬라이딩 평균
    ema_alpha: float = 0.5
    #ema = alpha * new_value + (1 - alpha) * old_ema    

@dataclass
class CerebellarNetConfig(SimulationConfig):
    n_grc: int = 512     # 모형화된 과립세포 수
    n_pkj: int = 64        # 모형화된 푸르키네 세포 수
    n_cf: int = 8         # 모형화된 교세포 수
