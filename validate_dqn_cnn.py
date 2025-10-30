# validate_dqn.py
import os, time, argparse, csv, math, random, collections
import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from config import SimulationConfig
from mujoco_model_v6 import Environment
from snn_model import SpikingCNN   # <-- 여기만 쓰면 됨
from datetime import datetime

cfg = SimulationConfig()

# --------------------------------------------------
# 디바이스
# --------------------------------------------------
def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# --------------------------------------------------
# 이 모델용 Q(=motor) 추출
# --------------------------------------------------
@torch.no_grad()
def q_values_from_model(model: nn.Module, obs_batch: torch.Tensor):
    """
    SpikingCNN은 forward에서 바로 (B,4) motor 같은 걸 내보내므로
    여기서는 그냥 한 번 forward만 하면 됨.
    에피소드 시작 때만 전체 net reset 했다고 가정.
    obs_batch: (B, 1, H, W) float32
    return: (B, A)
    """
    out = model(obs_batch)   # (B,4)
    # 혹시 (B,4,1,1) 이런 식으로 나오면 평탄화
    if out.ndim > 2:
        out = out.view(out.size(0), -1)
    return out

@torch.no_grad()
def select_action_greedy(model, obs, n_actions, device):
    """
    검증용: 탐험 없이 네트워크 출력 그대로 사용
    obs: (1, H, W) 또는 (1,1,H,W) 들어올 수 있음
    """
    # (1,1,H,W)로 맞추기
    if obs.ndim == 3:
        # (1,H,W) -> (1,1,H,W)
        obs_t = torch.from_numpy(obs).float().unsqueeze(1).to(device)
    else:
        # 이미 (1,1,H,W)
        obs_t = torch.from_numpy(obs).float().to(device)

    q = q_values_from_model(model, obs_t)  # (1, A)
    q = q.squeeze(0)                       # (A,)
    # 혹시 A가 더 크거나 작을 때 대비해서 잘라줌
    if q.numel() > n_actions:
        q = q[:n_actions]
    elif q.numel() < n_actions:
        pad = torch.zeros(n_actions - q.numel(), device=q.device)
        q = torch.cat([q, pad], dim=0)

    return q.cpu().numpy()                 # (A,)

# --------------------------------------------------
# 최신 pth 찾기
# --------------------------------------------------
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

    # 재현성
    if getattr(cfg, "seed", None) is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    device = get_device()

    # 환경
    env = Environment(cfg=cfg)
    n_actions = int(getattr(cfg, "n_motor", 4))

    # 모델
    model = SpikingCNN().to(device)
    model.eval()

    # 체크포인트 로드
    pth_path = load_latest_pth(pth_or_dir, trial_num)
    state = torch.load(pth_path, map_location=device)
    # 저장 형식에 따라 분기
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        # 그냥 state_dict만 저장해 둔 경우
        model.load_state_dict(state)
    else:
        # {"model": ..., "opt": ...} 이런 경우
        model.load_state_dict(state["model"])
    print(f"[VAL] Loaded: {pth_path}")

    # 로그 파일
    val_dir = os.path.join(DIR, f"{trial_num}_train")
    os.makedirs(val_dir, exist_ok=True)
    val_csv = os.path.join(val_dir, "VAL_LOG.csv")
    new_file = not os.path.exists(val_csv)

    # 최대 스텝
    max_steps_per_ep = int(cfg.simulation_duration_s / cfg.dt)

    rewards = []
    lengths = []
    durations = []

    for ep in range(1, episodes + 1):
        # 에피소드 시작 시점에만 SNN 리셋
        functional.reset_net(model)

        obs, _ = env.reset()
        # (H,W) -> (1,H,W)
        if obs.ndim == 2:
            obs = obs[None, ...]
        done = False
        ep_reward = 0.0
        steps = 0
        last_info = {}

        t0 = time.time()
        while not done and steps < max_steps_per_ep:
            steps += 1

            # 현재 obs로 행동 생성
            action_vec = select_action_greedy(model, obs, n_actions, device)  # (A,)

            # 환경 한 스텝
            next_obs, r, terminated, truncated, info = env.step(action_vec)
            done = bool(terminated or truncated)
            if info:
                last_info = info

            # obs 형태 맞추기
            if next_obs.ndim == 2:
                next_obs = next_obs[None, ...]
            ep_reward += float(r)
            obs = next_obs

            # 렌더링 옵션 있으면 여기서
            if render and hasattr(env, "render"):
                try:
                    env.render()
                except Exception:
                    pass

        duration = round(time.time() - t0, 3)

        # info에서 부가 정보
        bx = last_info.get("ball_pos_x", np.nan)
        by = last_info.get("ball_pos_y", np.nan)
        ball_mass = last_info.get("ball_mass", np.nan)

        # 시간 스탬프 (로컬)
        now_local = datetime.now()
        stamp = f"{now_local.month:02}/{now_local.day:02} {now_local.hour:02}:{now_local.minute:02}:{now_local.second:02}"

        # 콘솔
        print(f"[VAL] [Episode {ep}] R={ep_reward:.2f}, steps={steps}, duration={duration}s, datetime:{stamp}")

        # CSV 기록
        with open(val_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["episode", "datetime", "duration", "total_reward", "ball_x", "ball_y", "ball_mass_g"])
                new_file = False
            w.writerow([
                ep,
                stamp,
                duration,
                ep_reward,
                bx,
                by,
                (ball_mass * 1000) if not (isinstance(ball_mass, float) and np.isnan(ball_mass)) else ""
            ])

        rewards.append(ep_reward)
        lengths.append(steps)
        durations.append(duration)

        # 뷰어 닫기 시도
        if hasattr(env, "close_viewer"):
            try:
                env.close_viewer()
            except Exception:
                pass

    # 요약
    R_avg = float(np.mean(rewards)) if rewards else 0.0
    L_avg = float(np.mean(lengths)) if lengths else 0.0
    T_avg = float(np.mean(durations)) if durations else 0.0
    print(f"[VAL] Done. Episodes={episodes} | AvgR={R_avg:.3f} | AvgLen={L_avg:.1f} | AvgDur={T_avg:.3f}s")

    # 마지막 정리
    if hasattr(env, "close_viewer"):
        try:
            env.close_viewer()
        except Exception:
            pass

# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./runs", help="훈련 결과 디렉터리(DIR)")
    ap.add_argument("--trial", default="0001", help="trial_num")
    ap.add_argument("--pth", default="./runs", help="모델 .pth 경로 또는 DIR (최신 pth 자동 탐색)")
    ap.add_argument("--episodes", type=int, default=20, help="검증 에피소드 수")
    ap.add_argument("--render", action="store_true", help="뷰어 렌더링")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    validate(args.dir, args.trial, args.pth, args.episodes, args.render, args.debug)

if __name__ == "__main__":
    main()
