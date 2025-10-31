import math
import random
import sys, os
import numpy as np
from typing import List, Tuple

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from snn_model import SpikingCNN
from spikingjelly.activation_based import surrogate
from mujoco_model_v11 import Environment
from config import SimulationConfig
cfg = SimulationConfig()

# ============================================================
# 1) 정책 네트워크 (CPU 전용)
# ============================================================
class SpikingCNNPolicyWrapper(SpikingCNN):
    def __init__(self, T=None, log=False, init_log_std=-0.5):
        super().__init__(T=cfg.T if T is None else T, log=log)
        self.use_tanh_on_mu = False
        self.log_std = nn.Parameter(torch.ones(4) * init_log_std)

    def forward(self, mf: torch.Tensor):
        motor = super().forward(mf)  # (B, 4)

        mu = motor
        if self.use_tanh_on_mu:
            mu = torch.tanh(mu)

        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        return mu, std


class SymmetricQ(nn.Module):
    """
    train(): 원본만 forward
    eval(): 4방향(원본, 좌우, 상하, 둘다) 평균
    액션 순서: 0=+X, 1=-X, 2=+Y, 3=-Y
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.register_buffer("perm_lr",   torch.tensor([1, 0, 2, 3], dtype=torch.long))
        self.register_buffer("perm_ud",   torch.tensor([0, 1, 3, 2], dtype=torch.long))
        self.register_buffer("perm_both", torch.tensor([1, 0, 3, 2], dtype=torch.long))

    def _apply_sym(self, obs, kind: str):
        if kind == "orig":
            return self.base(obs)
        elif kind == "lr":
            q = self.base(torch.flip(obs, dims=[-1]))
            return q.index_select(1, self.perm_lr)
        elif kind == "ud":
            q = self.base(torch.flip(obs, dims=[-2]))
            return q.index_select(1, self.perm_ud)
        else:  # both
            q = self.base(torch.flip(obs, dims=[-2, -1]))
            return q.index_select(1, self.perm_both)

    def forward(self, obs):
        # 학습 중이면 원본만
        if self.training:
            return self._apply_sym(obs, "orig")

        # 평가 모드면 4개 평균
        q_orig = self._apply_sym(obs, "orig")
        q_lr   = self._apply_sym(obs, "lr")
        q_ud   = self._apply_sym(obs, "ud")
        q_both = self._apply_sym(obs, "both")
        return 0.25 * (q_orig + q_lr + q_ud + q_both)
# ============================================================
# 2) obs 를 무조건 CPU Tensor (1,1,10,10)로
# ============================================================
def to_obs_tensor(x):
    # gymnasium: (obs, info)
    if isinstance(x, tuple):
        x, _info = x

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    x = x.float()  # CPU

    # 모양 맞추기
    if x.dim() == 2:      # (10,10)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:    # (1,10,10)
        x = x.unsqueeze(0)
    # dim==4면 그대로
    return x


# ============================================================
# 3) rollout (CPU 전용)
# ============================================================
def collect_rollout(env, policy: SpikingCNNPolicyWrapper, rollout_size: int):
    obs_list = []
    act_list = []
    logp_list = []
    rew_list = []

    obs = env.reset()
    obs = to_obs_tensor(obs)

    for _ in range(rollout_size):
        # 여기서 grad 필요하니까 no_grad 절대 쓰지 마
        mu, std = policy(obs)                 # mu,std: requires_grad=True
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()                # (1,4)  ← sample은 괜찮아
        log_prob = dist.log_prob(action).sum(dim=-1)   # (1,) ← 이게 grad의 뿌리

        # env 는 numpy (4,) 필요 → 여기서만 끊어준다
        action_np = action.detach().cpu().numpy().reshape(-1)

        step_out = env.step(action_np)
        # gymnasium / gym 둘 다 처리
        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_out

        next_obs = to_obs_tensor(next_obs)

        obs_list.append(obs)
        act_list.append(action)
        logp_list.append(log_prob)
        rew_list.append(float(reward))

        obs = next_obs
        if done:
            obs = env.reset()
            obs = to_obs_tensor(obs)
            break

    obs_batch = torch.cat(obs_list, dim=0)          # (T,1,10,10)
    act_batch = torch.cat(act_list, dim=0)          # (T,4)
    logp_batch = torch.cat(logp_list, dim=0)        # (T,)
    rew_batch = torch.tensor(rew_list, dtype=torch.float32)
    return obs_batch, act_batch, logp_batch, rew_batch


# ============================================================
# 4) reward 변화 기반 loss
# ============================================================
def reward_change_policy_loss(logp: torch.Tensor, rewards: torch.Tensor):
    if rewards.shape[0] < 2:
        return torch.tensor(0.0)

    r_diff = rewards[1:] - rewards[:-1]   # (T-1,)
    logp_used = logp[:-1]                 # (T-1,)
    loss = -(r_diff.detach() * logp_used).mean()
    return loss


# ============================================================
# 5) 학습 루프 (CPU 고정)
# ============================================================
def train(
    epochs: int = 500,
    rollout_size: int = 16,   # 환경에서 한 번에 모을 길이
    update_every: int = 64,   # 이거 모이면 한 번 업데이트
    lr: float = 1e-3,
):
    env = Environment(cfg)
    policy = SpikingCNNPolicyWrapper()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # 임시 버퍼
    buf_logp = []
    buf_reward = []

    for epoch in range(epochs):
        # 1) 환경에서 rollout_size 만큼 행동
        obs_b, act_b, logp_b, rew_b = collect_rollout(env, policy, rollout_size)

        # 2) 버퍼에 누적
        buf_logp.append(logp_b)   # 각자 길이가 T_i
        buf_reward.append(rew_b)

        # 3) 길이 계산
        total_steps = sum(x.shape[0] for x in buf_reward)

        # 4) 64개 이상 모였으면 업데이트
        if total_steps >= update_every:
            # 전부 이어붙이기
            logp_cat = torch.cat(buf_logp, dim=0)    # (N,)
            rew_cat  = torch.cat(buf_reward, dim=0)  # (N,)

            loss = reward_change_policy_loss(logp_cat, rew_cat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 버퍼 비우기
            buf_logp = []
            buf_reward = []

            print(f"[update] steps={total_steps} loss={loss.item():.4f}")


if __name__ == "__main__":
    # CPU ONLY
    train()





'''import math
import random
import sys, os
import numpy as np
from typing import List, Tuple
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from snn_model import SpikingCNN
from spikingjelly.activation_based import surrogate
from mujoco_model_v11 import Environment
from config import SimulationConfig
cfg = SimulationConfig()

# =========================
# 1) 정책 네트워크
# 입력: (B, 1, 10, 10)
# 출력: (B, 4)  -> 연속 action mean
# =========================

class SpikingCNNPolicyWrapper(SpikingCNN):
    """
    SpikingCNN을 policy처럼 써서 (mu, std)를 뽑아주는 래퍼.
    입력: (B, 1, 10, 10)
    출력: mu: (B, 4), std: (B, 4)
    """
    def __init__(self, T=None, log=False, init_log_std=-0.5):
        super().__init__(T=cfg.T if T is None else T, log=log)
        self.use_tanh_on_mu = False
        self.log_std = nn.Parameter(torch.ones(4) * init_log_std)

    def forward(self, mf: torch.Tensor):
        motor = super().forward(mf)  # (B, 4)

        mu = motor
        if self.use_tanh_on_mu:
            mu = torch.tanh(mu)

        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        return mu, std


# =========================
# 3) rollout 함수
#   - rollout_size 만큼 (obs, action, log_prob, reward) 저장


def to_obs_tensor(x, device):
    # 1) gymnasium 스타일: (obs, info)
    if isinstance(x, tuple):
        x, _info = x  # info는 버리거나 로그로 저장
    # 2) numpy → tensor
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    # 3) 이미 tensor면 그대로
    x = x.to(device).float()
    # 4) 모양 맞추기: (1,1,10,10)
    if x.dim() == 3:          # (1,10,10) 이라면 배치 차원 추가
        x = x.unsqueeze(0)
    if x.dim() == 2:          # (10,10) 이라면 채널, 배치 둘 다 추가
        x = x.unsqueeze(0).unsqueeze(0)
    return x

# =========================
@torch.no_grad()
def collect_rollout(env, policy: SpikingCNNPolicyWrapper, rollout_size: int, device: torch.device):
    obs_list: List[torch.Tensor] = []
    act_list: List[torch.Tensor] = []
    logp_list: List[torch.Tensor] = []
    rew_list: List[float] = []

    obs = env.reset()#.to(device).float()
    obs = to_obs_tensor(obs, device)

    for _ in range(rollout_size):
        mu, std = policy(obs)              # mu: (1,4), std: (4,)
        dist = Normal(mu, std)
        action = dist.sample()             # (1,4)
        log_prob = dist.log_prob(action).sum(dim=-1)   # (1,) → 한 스텝의 로그확률

        next_obs, reward, terminated, truncated, info = env.step(action.cpu()[0])
        done = bool(terminated or truncated)
        next_obs = next_obs#.to(device).float()

        obs_list.append(obs)
        act_list.append(action)
        logp_list.append(log_prob)
        rew_list.append(reward)

        obs = next_obs
        if done:
            break

    # tensor 로 묶어주기
    obs_batch = torch.cat(obs_list, dim=0)       # (T, 1, 10, 10)
    act_batch = torch.cat(act_list, dim=0)       # (T, 4)
    logp_batch = torch.cat(logp_list, dim=0)     # (T,)
    rew_batch = torch.tensor(rew_list, device=device, dtype=torch.float32)  # (T,)

    return obs_batch, act_batch, logp_batch, rew_batch


# =========================
# 4) reward 변화 기반 loss
#   - r_diff[t] = r[t] - r[t-1]
#   - r가 증가한 액션은 강화, 감소한 액션은 약화
#   - policy gradient 스타일: L = - Σ r_diff[t] * logπ(a_{t-1} | s_{t-1})
#   - 주의: r_diff는 T-1개, logp는 T개니까 맞춰줘야 함
# =========================
def reward_change_policy_loss(logp: torch.Tensor, rewards: torch.Tensor):
    """
    logp: (T,)
    rewards: (T,)
    """
    if rewards.shape[0] < 2:
        # rollout 길이가 1이면 학습할 게 없음
        return torch.tensor(0.0, device=rewards.device)

    r_diff = rewards[1:] - rewards[:-1]          # (T-1,)
    # logp도 한 칸 당겨서 맞춰준다: action_t → r_{t+1} - r_{t}
    logp_used = logp[:-1]                        # (T-1,)

    # advantage처럼 사용
    loss = -(r_diff.detach() * logp_used).mean()
    return loss


# =========================
# 5) 학습 루프 예시
# =========================
def train(
    epochs: int = 500,
    rollout_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
):
    device = torch.device(device)
    env = Environment(cfg)
    policy = SpikingCNNPolicyWrapper().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        policy.train()
        obs_b, act_b, logp_b, rew_b = collect_rollout(env, policy, rollout_size, device)

        loss = reward_change_policy_loss(logp_b, rew_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            avg_rew = rew_b.mean().item()

        if (epoch + 1) % 20 == 0:
            print(f"[{epoch+1:04d}] loss={loss.item():.4f}  avg_reward={avg_rew:.4f}")


if __name__ == "__main__":
    # xpu, cuda 있으면 여기서 선택해서 넘기면 됨
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train(device=dev)
  '''