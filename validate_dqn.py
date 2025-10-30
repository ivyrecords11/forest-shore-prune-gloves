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
from mujoco_model_v6 import Environment
from snn_model import CerebellarCNNAC2, SpikingCNN       # <- CerebellarCNNAC2 정의가 있는 모듈로 교체
from datetime import datetime

cfg = SimulationConfig()
# --------------------------------------------------
# 유틸
# --------------------------------------------------
def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def q_values_from_model(model, obs_batch):
    """
    obs_batch: (B, 1, H, W) float32
    actor의 마지막 LIF 전위를 Q(s)로 사용 (clip/tanh 없음)
    """
    model.actor.train(False)
    functional.reset_net(model.actor)
    for _ in range(model.T):
        _ = model.actor(obs_batch)
    q = model.actor[-1].v  # (B, A)
    return q

@torch.no_grad()
def select_action_greedy(model, obs, n_actions, device):
    """
    검증: 탐험 없이 네트워크 출력 그대로 사용 (연속 4값 컨트롤러)
    """
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)  # (1,1,H,W)
    q = q_values_from_model(model, obs_t).squeeze(0)               # (A,)
    return q.cpu().numpy()                                         # (A,)

def load_latest_pth(pth_path_or_dir, trial_num):
    """
    - 파일 경로가 오면 그대로 반환
    - 디렉터리가 오면 해당 trial 디렉터리에서 가장 최신 pth 선택
    """
    if os.path.isfile(pth_path_or_dir):
        return pth_path_or_dir

    # 디렉터리 입력인 경우
    cand_dir = pth_path_or_dir
    # 기본 학습 스크립트의 저장 규칙: {DIR}/{trial_num}_train/model_params_*.pth
    trial_dir = os.path.join(cand_dir, f"{trial_num}_train")
    if not os.path.isdir(trial_dir):
        raise FileNotFoundError(f"trial dir not found: {trial_dir}")

    cands = [os.path.join(trial_dir, f) for f in os.listdir(trial_dir)
             if f.endswith(".pth") and f.startswith("model_params_")]
    if not cands:
        raise FileNotFoundError(f"No .pth files in {trial_dir}")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

# --------------------------------------------------
# 검증 루프
# --------------------------------------------------
def validate(DIR: str, trial_num: str, pth_or_dir: str,
             episodes: int, render: bool, DEBUG: bool):
    cfg = SimulationConfig()

    # 재현성(가능하면)
    if getattr(cfg, "seed", None) is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    device = get_device()

    # 환경 구성: cfg를 최대한 넘겨줌
    env = Environment(cfg=cfg)  # 필요에 따라 인자 조정
    n_actions = int(getattr(cfg, "n_motor", 4))

    # 모델 구성 (cfg 기반)
    model = SpikingCNN().to(device)
    model.eval()

    # 체크포인트 로드
    pth_path = load_latest_pth(pth_or_dir, trial_num)
    state = torch.load(pth_path, map_location=device)
    # 학습 스크립트가 state_dict만 저장했으므로 그대로 로드
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        model.load_state_dict(state)
    else:
        # 만약 전체 dict로 저장했다면 안전 로드
        model.load_state_dict(state["model"])
    print(f"[VAL] Loaded: {pth_path}")

    # 검증 로그 파일
    val_dir = os.path.join(DIR, f"{trial_num}_train")
    os.makedirs(val_dir, exist_ok=True)
    val_csv = os.path.join(val_dir, "VAL_LOG.csv")
    new_file = not os.path.exists(val_csv)

    # 한 에피소드 최대 스텝
    max_steps_per_ep = int(cfg.simulation_duration_s / cfg.dt)

    rewards = []
    lengths = []
    durations = []

    for ep in range(1, episodes + 1):
        functional.reset_net(model)
        obs, _ = env.reset()
        if obs.ndim == 2:
            obs = obs[None, ...]  # (1,H,W)

        done = False
        ep_reward = 0.0
        steps = 0
        last_info = {}

        t0 = time.time()
        while not done and steps < max_steps_per_ep:
            steps += 1
            action_vec = select_action_greedy(model, obs, n_actions, device)  # (A,)
            next_obs, r, terminated, truncated, info = env.step(action_vec)
            done = bool(terminated or truncated)
            last_info = info or last_info
            if next_obs.ndim == 2:
                next_obs = next_obs[None, ...]
            ep_reward += float(r)
            obs = next_obs

        duration = round(time.time() - t0, 3)

        # info에서 부가 정보 추출(없으면 NaN)
        bx = last_info.get("ball_pos_x", np.nan)
        by = last_info.get("ball_pos_y", np.nan)
        ball_mass = last_info.get("ball_mass", np.nan)

        # 시간 스탬프
        now_utc = datetime.now()
        stamp = f"{now_utc.month:02}/{now_utc.day:02} {now_utc.hour:02}:{now_utc.minute:02}:{now_utc.second:02}"

        # 콘솔 출력
        print(f"[VAL] [Episode {ep}] R={ep_reward:.2f}, steps={steps}, duration={duration}s, datetime:{stamp}")

        # CSV 기록
        with open(val_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["episode", "datetime", "duration", "total_reward", "ball_x", "ball_y", "ball_mass_g"])
                new_file = False
            w.writerow([ep, stamp, duration, ep_reward, bx, by,
                        (ball_mass*1000) if not (isinstance(ball_mass, float) and np.isnan(ball_mass)) else ""])

        rewards.append(ep_reward)
        lengths.append(steps)
        durations.append(duration)
        env.close_viewer()

    # 요약
    R_avg = float(np.mean(rewards)) if rewards else 0.0
    L_avg = float(np.mean(lengths)) if lengths else 0.0
    T_avg = float(np.mean(durations)) if durations else 0.0
    print(f"[VAL] Done. Episodes={episodes} | AvgR={R_avg:.3f} | AvgLen={L_avg:.1f} | AvgDur={T_avg:.3f}s")

    # 뷰어 정리(있는 경우만)
    close_viewer = getattr(env, "close_viewer", None)
    if callable(close_viewer):
        try: close_viewer()
        except Exception:
            pass

# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./runs", help="훈련 결과 디렉터리(DIR)")
    ap.add_argument("--trial", default="0001", help="trial_num")
    ap.add_argument("--pth", default="./runs", help="모델 .pth 파일 경로나, DIR 디렉터리(최신 pth 자동 탐색)")
    ap.add_argument("--episodes", type=int, default=20, help="검증 에피소드 수")
    ap.add_argument("--render", action="store_true", help="뷰어 렌더링")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    validate(args.dir, args.trial, args.pth, args.episodes, args.render, args.debug)
    
"""
# 최신 pth 자동 탐색해서 검증
python validate_dqn.py --dir ./runs --trial 0001 --pth ./runs --episodes 20

# 특정 체크포인트로 검증
python validate_dqn.py --dir ./runs --trial 0001 --pth TRAIN_DQN_CNN\0_train\model_csv\model_params_1270.csv --episodes 10 --render

"""

if __name__ == "__main__":
    main()
