# test_pd_controller.py
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import os

import torch
from config import SimulationConfig
from mujoco_model_v3 import Environment  # 위 파일 이름에 맞춰 변경
from utils_logger import SpikeHeatmap, SpikePlotter

def pd_controller(e, e_prev, dt, Kp, Kd, u_clip=1.0):
    # 이산형 PD: u = Kp*e + Kd*(e - e_prev)/dt
    de = (e - e_prev) / max(dt, 1e-9)
    u  = Kp * e + Kd * de
    # [-u_clip, +u_clip]로 포화
    '''
    if isinstance(u, np.ndarray):
        return np.clip(u, -u_clip, u_clip), de
    else:
        return max(-u_clip, min(u, u_clip)), de
    '''
    return u, de

def to_four_motor(u_x, u_y):
    # 연속 제어를 4채널 [XP, XN, YP, YN] (각 [0,1])로 변환
    return np.array([
        max(u_x, 0.0),  # XP
        max(-u_x, 0.0), # XN
        max(u_y, 0.0),  # YP
        max(-u_y, 0.0)  # YN
    ], dtype=np.float32)

def make_cfg():
    # 네 설정에 맞춰 필요값만 예시로 지정
    return SimulationConfig(
        plate_size=0.30,
        n_sensor_1d=10,
        n_mf=100,
        n_motor=4,
        dt=1e-3,                    # 1 ms
        ball_mass=None,             # reset 때 무작위
        ball_density=1000.0,
        simulation_duration_s=10.0,
        success_duration_s=0.5,
        success_reward_per_sec=10000,
        penalize_failure_per_sec=10000,
        transmitt_speed=50.0,
        motor_max_hinge_deg=15.0, 
        motor_gain = 1.0
    )

def run_pd_test():
    cfg = make_cfg()
    env = Environment(cfg, render=True)
    import os

    # Ensure directory exists
    os.makedirs("./pd_controller_test", exist_ok=True)
    plotter = SpikePlotter(plot_name="./pd_controller_test/spike_plot")
    plotter_action = SpikePlotter(plot_name="./pd_controller_test/action_plot")
    heatmap = SpikeHeatmap(plot_name="./pd_controller_test/spike_heatmap",shape=(cfg.n_sensor_1d,cfg.n_sensor_1d))

    obs, info = env.reset()
    dt = cfg.dt
    alpha = 3
    beta = 15
    # PD 게인(보수적으로 시작해서 튠)
    # 위치 범위가 ±0.15 m 정도이므로, Kp*|e|max ~ 1 근방이 되게 시작
    Kp_x, Kd_x = 4.0*alpha, 0.4*alpha*beta
    Kp_y, Kd_y = 4.0*alpha, 0.4*alpha*beta
    # 제어 신호 포화(환경은 [-1,1] 차동에 max_radians를 곱하므로 여기선 정규화 축)
    U_CLIP = 1.0

    e_prev_x = -env.ball_x
    e_prev_y = -env.ball_y

    ep_reward = 0.0
    for t in range(int(cfg.simulation_duration_s / dt)):
        # 현재 위치 -> 에러 (목표 0,0)
        e_x = -env.ball_x
        e_y = -env.ball_y

        # 이산 PD
        u_x, de_x = pd_controller(e_x, e_prev_x, dt, Kp_x, Kd_x, u_clip=U_CLIP)
        u_y, de_y = pd_controller(e_y, e_prev_y, dt, Kp_y, Kd_y, u_clip=U_CLIP)

        # 4채널 액션으로 변환
        action = to_four_motor(u_x, u_y)

        # 환경 진행
        obs, reward, terminated, truncated, info = env.step(action)
        plotter.save_spikes(obs)
        plotter_action.save_spikes(action)
        heatmap.save_spikes(obs)

        ep_reward += reward

        # 로그(원하면 주기 늘리기)
        if t % 500 == 0:
            dist = math.hypot(env.ball_x, env.ball_y)
            print(f"[{t:5d}] e=({e_x:+.4f},{e_y:+.4f}) u=({u_x:+.3f},{u_y:+.3f}) "
                  f"pos=({env.ball_x:+.4f},{env.ball_y:+.4f}) dist={dist:.4f} R={reward:.2f}")

        # 종료 조건
        if terminated or truncated:
            print(f"Episode end at t={t}, total_reward={ep_reward:.2f}, "
                  f"terminated={terminated}, truncated={truncated}")
            break

        # 다음 스텝 준비
        e_prev_x, e_prev_y = e_x, e_y
        #time.sleep(0.0006)

    # 뷰어 사용 시 약간 대기(선택)
    time.sleep(0.2)
    #plotter.plot(visualize=True)
    plotter_action.plot(visualize=True)
    heatmap.plot(visualize=True)

if __name__ == "__main__":
    run_pd_test()