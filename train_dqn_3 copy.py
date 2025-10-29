# dqn_train_cfg.py


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
from mujoco_model_v11 import Environment
from snn_model import CerebellarCNNAC2# <- CerebellarCNNAC2 정의가 있는 모듈로 교체

from utils_logger import SpikeHeatmap, SpikePlotter, PotentialPlotter
cfg = SimulationConfig()

# =========================
# 1) CFG 로드 & 시드 고정
# =========================
if cfg.seed is not None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

# =========================
# 2) 디바이스 선택
# =========================
def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# =========================
# 3) 리플레이 버퍼
# =========================
Transition = collections.namedtuple('Transition', ('s', 'a', 'r', 'ns', 'd'))

class ReplayBuffer:
    def __init__(self, capacity:int):
        self.buf = collections.deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch_size:int):
        batch = random.sample(self.buf, batch_size)
        s  = torch.stack(([b.s  for b in batch]), dim=0)
        a  = torch.tensor(np.array([b.a  for b in batch]), dtype=torch.float32, device=s.device)
        r  = torch.tensor(np.array([b.r  for b in batch]), dtype=torch.float32, device=s.device)
        ns = torch.stack(([b.ns for b in batch]), dim=0)
        d  = torch.tensor(np.array([b.d  for b in batch]), dtype=torch.float32, device=s.device)
        return s, a, r, ns, d
    def __len__(self): return len(self.buf)

# =========================
# 4) Q값(=actor 마지막 전위) 계산 유틸
# =========================
@torch.no_grad()
def q_values_from_model(model, obs_batch):
    # obs_batch: (B, 1, H, W)
    assert obs_batch.shape == (1,1,10,10), f"[!]obs_batch:{obs_batch}"
    model.train(False)
    #functional.reset_net(model.actor)
    for _ in range(model.T):
        #action = model.actor(obs_batch)
        #print(obs_batch, action)
        q = model.forward(obs_batch)
    assert q[-1].shape == torch.Size([4]), f"q[0]: {q[0]}, q: {q}"
    #print("==========================================================")
    #print(action)
    return q[-1] # (B, n_actions)

# =========================
# 5) DQN 최적화 스텝
# =========================
def dqn_optimize(model, target, buffer, optimizer, batch_size, gamma, max_grad_norm, device):
    """
    Multi-output DQN update for vector controllers (e.g., 4 outputs).
    Trains all A dimensions at once with elementwise TD targets:
        y = r + gamma*(1-d)*q_next_target
    Loss = SmoothL1(q_s, y) over all elements.
    """
    if len(buffer) < batch_size:
        return None

    model.train(True)

    # s: (B, 1, H, W), a: (B, A) [unused here], r: (B,), ns: (B, 1, H, W), d: (B,)
    s, a, r, ns, d = buffer.sample(batch_size)
    s, ns = s.to(device), ns.to(device)
    r, d  = r.to(device), d.to(device)

    # --- Q(s):  T steps, use actor membrane potential as Q
    #functional.reset_net(model.actor)
    for _ in range(model.T):
        _ = model.forward(s)
    q_s = model.lif_out_actor.v  # (B, A), no tanh/no clip

    # --- Target Q(ns): elementwise TD target (no action argmax)
    with torch.no_grad():
        #functional.reset_net(target.actor)
        for _ in range(target.T):
            _ = target.forward(ns)
        q_ns_target = target.lif_out_actor.v  # (B, A)

        # y = r + gamma*(1-d)*q_ns_target, broadcast r,d to (B, A)
        y = r.unsqueeze(1) + gamma * (1.0 - d).unsqueeze(1) * q_ns_target  # (B, A)

    # --- Loss over all elements
    loss = torch.nn.functional.smooth_l1_loss(q_s, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm is not None:
        from torch.nn.utils import clip_grad_norm_
        clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return float(loss.item())

# =========================
# 6) 타깃 네트 업데이트
# =========================
def soft_update(target, online, tau=0.01):
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.data.lerp_(op.data, tau)

def hard_update(target, online):
    target.load_state_dict(online.state_dict())

# =========================
# 7) ε-탐욕 정책
# =========================

def select_action_epsilon_greedy(model, obs, eps, n_actions, device):
    """
    ε-greedy for multi-controller (e.g., 4 outputs).
    Returns the raw Q-value vector (no clipping, no tanh).

    - With probability ε, replace ONE random element with a random value.
    - Otherwise, use the model output as-is.
    """
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)  # (1,1,H,W)
        assert obs_t.shape==(1,1,10,10), "[!] shape이 맞지 않음."
        q_values = q_values_from_model(model, obs_t).squeeze(0)        # (n_actions,)
        assert q_values.shape == torch.Size([n_actions]), f"q_values.shape은 {q_values.shape}임"
        # obs_batch도 가능, 근데 단일임.
    # Convert to numpy for env
    action = q_values.clone()

    if random.random() < eps:
        rand_idx = random.randint(0, n_actions - 1)
        rand_val = random.uniform(float(q_values.min()), float(q_values.max()))
        action[rand_idx] = rand_val
        # optional debug:         print(f"[ε-greedy] Randomized controller[{rand_idx}] = {rand_val:.3f}")

    return action.cpu().numpy()  # shape: (n_actions,)

# =========================
# 8) (선택) 가중치 초기화: cfg.weight_init 반영
# =========================
def apply_weight_init_from_cfg(model, weight_init: str):
    wi = (weight_init or "").lower()
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if "kaiming" in wi:
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            elif "xavier" in wi:
                nn.init.xavier_normal_(m.weight, gain=1.0)
            elif "normal-0" in wi:
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
            # bias는 없음(질문 코드 기준 bias=False), 있으면 0으로
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

# =========================
# 9) 학습 루프
# =========================
def train_dqn(env, model, *,
              episodes: int = None,
              gamma: float = 0.1,
              max_grad_norm: float = 1.0,
              target_update_mode: str = "soft",   # "soft" | "hard"
              tau: float = 0.01,
              hard_update_interval: int = 1000,
              DIR: str = "./runs",                 # <-- 추가
              trial_num: int = 1,             # <-- 추가
              start_epoch: int = 0,                # <-- 추가
              DEBUG: bool = True):                 # <-- 추가

    import time, csv
    from datetime import datetime
    os.makedirs(f"{DIR}/{trial_num}_train", exist_ok=True)

    device = get_device()

    # ---- cfg 활용 ----
    lr = float(cfg.eta)
    T  = int(cfg.T)
    model.T = T
    max_steps_per_ep = int(cfg.simulation_duration_s / cfg.dt)
    if start_epoch < 2:
        start_learning = 30000
    else: 
        start_learning = max(cfg.num_steps, 512)
    train_freq = 4
    batch_size = cfg.num_steps
    buffer_size = min(max_steps_per_ep * 50, 50000)

    if episodes is None:
        episodes = 50
    max_epochs = episodes  # 요청 코드와 호환

    apply_weight_init_from_cfg(model, cfg.weight_init)

    from copy import deepcopy
    target = deepcopy(model).to(device)
    target.eval()
    hard_update(target, model)

    optim_params = [p for n, p in model.named_parameters() if "log_std" not in n]
    optimizer = optim.Adam(optim_params, lr=lr)

    rb = ReplayBuffer(buffer_size)
    n_actions = 4

    eps_start, eps_end = cfg.epsilon_start, cfg.epsilon_end
    eps_decay_steps = max(10_000, cfg.num_steps * 100)
    eps = eps_start
    eps_decay = (eps_start - eps_end) / eps_decay_steps

    model.to(device)

    global_step = 0
    for ep in range(episodes):
        # ---- 10회마다 로깅 1회만 활성화 ----
        log_this = ((ep) % 10 == 0)
        model.set_logging(log_this)
        if log_this:
            logger_heatmap = SpikeHeatmap(plot_name=f"{DIR}/{trial_num}_train/plots/sensor_heatmap_epoch_{ep}", shape=(cfg.n_sensor_1d, cfg.n_sensor_1d))
            logger_output_v = PotentialPlotter(plot_name=f"{DIR}/{trial_num}_train/plots/output_spike_epoch_{ep}")
        else:
            logger_heatmap = None
            logger_output_v = None

        # ---- 에피소드 시작 ----
        functional.reset_net(model)
        t0 = time.time()
        obs, info_init = env.reset()
        if obs.ndim == 2:
            obs = obs[None, ...]  # (1,H,W)

        done, ep_reward, steps = False, 0.0, 0
        last_info = {}

        while not done and steps < max_steps_per_ep:
            global_step += 1
            steps += 1
            eps = max(eps_end, eps - eps_decay)
            
            a = select_action_epsilon_greedy(model, obs, eps, n_actions, device)
            
            #log
            if log_this:
                logger_heatmap.save_spikes(obs)
                logger_output_v.save_spikes(a)
                
            next_obs, r, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            last_info = info or last_info
            if next_obs.ndim == 2:
                next_obs = next_obs[None, ...]
            ep_reward += float(r)

            s_t  = torch.from_numpy(obs).float()
            ns_t = torch.from_numpy(next_obs).float()
            rb.push(s_t, a, float(r), ns_t, float(done))

            obs = next_obs

            if global_step > start_learning and (global_step % train_freq == 0):
                _ = dqn_optimize(model, target, rb, optimizer, batch_size,
                                 gamma, max_grad_norm, device)

            if target_update_mode == "soft":
                soft_update(target, model, tau)
            else:
                if global_step % hard_update_interval == 0:
                    hard_update(target, model)

        # ===== 여기부터: 요청한 저장/로그 코드로 교체 =====
        if log_this:
            logger_output_v.plot()
            logger_heatmap.plot()
            logger_output_v.clear()
            logger_heatmap.clear()
            del logger_output_v, logger_heatmap
        functional.reset_net(model)
        duration = steps
        total_reward = ep_reward
        entropy = float("nan")   # DQN에서는 의미 없음. 필요시 0.0로 대체 가능

        # ball_pos, ball_mass 안전 추출
        ball_pos = info_init.get("ball_pos",np.nan)
        ball_mass = info_init.get("ball_mass", np.nan)

        now_utc = datetime.now()
        month = now_utc.month
        day = now_utc.day
        hour = now_utc.hour
        minute = now_utc.minute
        second = now_utc.second

        epoch = start_epoch + ep + 1
        print(f"[TRAIN] [Epoch {epoch}] total_reward: {total_reward:.2f}, duration: {duration}, entropy: {entropy:.4f}, datetime:{month:02}/{day:02} {hour:02}:{minute:02}:{second:02}")
        print(f"[TRAIN] Total Reward/Distance(L1) from Center:{total_reward/abs(ball_pos[0]+ball_pos[1])}")
        log_csv = f"{DIR}/{trial_num}_train/LOG.csv"
        new_file = not os.path.exists(log_csv)
        with open(log_csv, "a", newline="") as file_train_log:
            if new_file:
                file_train_log.write("epoch,datetime,duration,total_reward,ball_x,ball_y,ball_mass_g\n")
            file_train_log.write(
                f"{epoch},{month:02}/{day:02} {hour:02}:{minute:02}:{second:02},{duration},{total_reward},{ball_pos[0]},{ball_pos[1]},{(ball_mass*1000) if not np.isnan(ball_mass) else ''}\n"
            )

        # storage.clear() (있을 때만)
        storage = globals().get("storage", None)
        if storage is not None and hasattr(storage, "clear"):
            storage.clear()

        # env.close_viewer() (있을 때만)
        close_viewer = getattr(env, "close_viewer", None)
        if callable(close_viewer):
            try: close_viewer()
            except Exception: pass

        #if DEBUG: print("[TRAIN] Training complete.")

        # 모델 가중치(pth)
        pth_path = f"{DIR}/{trial_num}_train/model_params_{epoch}.pth"
        torch.save(model.state_dict(), pth_path)

        # (디버깅용) state_dict 프린트

        # CONFIG 저장
        with open(f"{DIR}/{trial_num}_train/CONFIG.txt", "w", encoding="utf-8") as file_config:
            file_config.write(str(cfg))

        # 모델 파라미터 CSV 저장(요청한 중첩 루프 형태 그대로)
        with open(f"{DIR}/{trial_num}_train/model_params_{epoch}.csv", "a", newline="") as file_params:
            writer = csv.writer(file_params)
            writer.writerow(["Layer", "Parameter Name", "Values"])
            for name, param in model.state_dict().items():
                file_params.write(f"\n{name.split('.')[0]},{name}\n")
                arr = param.detach().cpu().numpy()
                # 원본 로직 유지(중첩 차원 펼치기)
                if arr.ndim >= 1:
                    for items in arr:
                        if getattr(items, "ndim", 0) >= 1:
                            for it in items:
                                if getattr(it, "ndim", 0) >= 1:
                                    for it2 in it:
                                        if getattr(it2, "ndim", 0) >= 1:
                                            for it3 in it2:
                                                file_params.write(f"{it3},")
                                        else:
                                            file_params.write(f"{it2}")
                                        file_params.write("\n")
                                else:
                                    file_params.write(f"{it}\n")
                        else:
                            file_params.write(f"{items}\n")
                else:
                    file_params.write(f"{arr}\n")

        # MANIFEST 업데이트
        with open(f"{DIR}/MANIFEST.csv", "a") as file_manifest:
            file_manifest.write(f"{trial_num},{total_reward},{cfg}\n")

        #if DEBUG: print("[TRAIN] All data saved.")
        # ===== 교체 끝 =====

        # 에피소드 종료 정리
        if model.log:
            try:
                model.clear_monitor()
            except Exception:
                pass
        os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == '__main__':
    # -------------------------------
    # 0. Create main directory
    # -------------------------------
    DIR = "./TRAIN_DQN"
    os.makedirs(DIR, exist_ok=True)

    # -------------------------------
    # 3. Initialize environment/model
    # -------------------------------
    env = Environment(cfg=cfg)
    model = CerebellarCNNAC2()

    # -------------------------------
    # 1. Prepare MANIFEST and trial
    # -------------------------------
    start_epoch = 0
    trial_num = 0

    manifest_path = f"{DIR}/MANIFEST.csv"
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w") as f:
            f.write("trial_id,total_reward,config\n")
        print("[TRAIN] MANIFEST.csv initialized.")
    else:
        print("[TRAIN] initializing from MANIFEST.csv")

    with open(manifest_path, "r") as f:
        lines = f.readlines()
        last_trial_line = lines[-1].strip() if len(lines) > 1 else None

    if last_trial_line:
        print(f"[TRAIN] Last trial found: {last_trial_line}")
        c = input("[TRAIN] Continue on last trial? (y/n): ")
        if c.lower() == "y":
            trial_num = int(last_trial_line.split(",")[0])
            # recover start epoch
            log_path = f"{DIR}/{trial_num}_train/LOG.csv"
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    start_epoch = int(rows[-1][0]) + 1 if len(rows) > 1 else 0
            model_pth = f"{DIR}/{trial_num}_train/model_params_{start_epoch-1}.pth"
            if os.path.exists(model_pth):
                print(f"[TRAIN] Loading model from {model_pth}")
                model.load_state_dict(torch.load(model_pth))
            else:
                print(f"[WARN] No checkpoint found at {model_pth}. Starting fresh.")
        elif c.lower() == "n":
            trial_num = int(last_trial_line.split(",")[0]) + 1
        else:
            print("[TRAIN] Invalid input. Exiting.")
            exit(1)
    else:
        trial_num = 0
        print("[TRAIN] No previous trial found; starting new one.")

    # -------------------------------
    # 2. Prepare directories
    # -------------------------------
    dir_path = f"{DIR}/{trial_num}_train"
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(os.path.join(dir_path, "plots"), exist_ok=True)

    print(f"\n[TRAIN] Trial: {trial_num}")
    print(f"[TRAIN] Config: {cfg}")

    # -------------------------------
    # 4. Start training
    # -------------------------------
    train_dqn(
        env,
        model,
        DIR=DIR,
        trial_num=trial_num,
        start_epoch=start_epoch,
        DEBUG=True
    )
