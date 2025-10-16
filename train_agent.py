import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple
from mujoco_model_v3 import Environment
from config import SimulationConfig

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# SpikingJelly (activation_based)
from spikingjelly.activation_based import neuron, surrogate, functional


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # cerebellar dims
    n_goc: int = 8
    n_grc: int = 16
    n_mli: int = 16
    n_pkj: int = 4
    n_cf: int = 16

    # motor/action dim will be set from env.action_space.n
    n_motor: int = 2

    # LIF time constants
    t_goc: float = 2.0
    t_grc: float = 2.0
    t_mli: float = 2.0
    t_pkj: float = 2.0
    t_motor: float = 2.0
    th_pkj: float = 4.0

    # encoder
    obs_dim: int = 4       # will be overridden from env.observation_space.shape[0]
    cf_dim: int = 16

    # R-STDP hyperparams
    A_plus: float = 0.01
    A_minus: float = 0.012
    tau_pre: float = 20.0
    tau_post: float = 20.0
    wmin: float = -0.2
    wmax: float = 0.2
    reward_scale: float = 0.05
    ema_reward_beta: float = 0.99

    # training
    episodes: int = 100
    max_steps: int = 500
    eps_greedy: float = 0.05
    seed: int = 7


# -----------------------------
# Simple Non-spiking LIF wrapper to expose v (membrane)
# -----------------------------
class NonSpikingLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def reset(self): 
        self.v = torch.tensor(0.0)

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)

        if self.training:
            self.neuronal_charge(x)
        else:
            if self.v_reset is None:
                if self.decay_input:
                    self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
                else:
                    self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)
                
            else:
                if self.decay_input:
                    self.v = self.neuronal_charge_decay_input(x, self.v, self.v_reset, self.tau)
                else:
                    self.v = self.neuronal_charge_no_decay_input(x, self.v, self.v_reset, self.tau)
        return self.v


# -----------------------------
# Observation Encoder: obs -> mf(1,10,10), cf(n_cf)
# -----------------------------
class ObsEncoder(nn.Module):
    def __init__(self, obs_dim: int, n_cf: int):
        super().__init__()
        self.obs2cf = nn.Linear(obs_dim, n_cf, bias=False)
        nn.init.xavier_uniform_(self.obs2cf.weight)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs: (N, obs_dim)
        cf = torch.tanh(self.obs2cf(obs))     # (N,n_cf)
        return cf


# -----------------------------
# Cerebellar network (with pooling goc->4x4 and pkj->2x2)
# -----------------------------
class CerebellarNet_old(nn.Module):
    def __init__(self, cfg: Config, action_dim: int):
        super().__init__()
        # Nodes
        self.goc = neuron.LIFNode(tau=cfg.t_goc, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.grc = neuron.LIFNode(tau=cfg.t_grc, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.mli = neuron.LIFNode(tau=cfg.t_mli, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.pkj = neuron.LIFNode(tau=cfg.t_pkj, v_threshold=cfg.th_pkj, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.motor = NonSpikingLIFNode(tau=cfg.t_motor)

        # Convs
        self.mf2goc  = nn.Conv2d(1,        cfg.n_goc, kernel_size=3, stride=1, padding=0, bias=False)  # 10->8
        self.mf2grc  = nn.Conv2d(1,        cfg.n_grc, kernel_size=5, stride=1, padding=0, bias=False)  # 10->6
        self.goc2grc = nn.Conv2d(cfg.n_goc, cfg.n_grc, kernel_size=3, stride=1, padding=0, bias=False) # 8->6
        self.grc2mli = nn.Conv2d(cfg.n_grc, cfg.n_mli, kernel_size=3, stride=1, padding=0, bias=False) # 6->4
        self.grc2pkj = nn.Conv2d(cfg.n_grc, cfg.n_pkj, kernel_size=3, stride=1, padding=0, bias=False) # 6->4
        self.mli2pkj = nn.Conv2d(cfg.n_mli, cfg.n_pkj, kernel_size=1, stride=1, padding=0, bias=False) # 4->4
        #self.cf2pkj  = nn.Linear(cfg.n_cf, cfg.n_pkj * 4 * 4, bias=False)

        # Heads — pkj pooled to 2x2
        self.pkj2motor = nn.Linear(cfg.n_pkj * 2 * 2, action_dim, bias=False)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.1, 0.01)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 1.0, 0.05)

    def forward(self, mf: torch.Tensor) -> Dict[str, torch.Tensor]:#, cf: torch.Tensor
        # goc branch
        goc = self.goc(self.mf2goc(mf))                      # (N, n_goc, 8, 8)
        _goc_4x4 = F.avg_pool2d(goc, 2, 2)                   # (N, n_goc, 4, 4)

        # grc/mli
        grc = self.grc(self.mf2grc(mf) - self.goc2grc(goc))  # (N, n_grc, 6, 6)
        mli = self.mli(self.grc2mli(grc))                    # (N, n_mli, 4, 4)

        # pkj maps @ 4x4
        pkj_pre = self.grc2pkj(grc) - self.mli2pkj(mli) #- self.cf2pkj(cf).view(cf.shape[0], -1, 4, 4)
        pkj = self.pkj(pkj_pre)                               # (N, n_pkj, 4, 4)
        pkj_2x2 = F.avg_pool2d(pkj, 2, 2)                     # (N, n_pkj, 2, 2)

        # motor head (non-spiking LIF to expose membrane v as preference)
        print(pkj_2x2.shape)
        flat = pkj_2x2.flatten()
        #motor_out = self.motor(self.pkj2motor(flat))          # (N, action_dim)
        motor_y = self.motor(self.pkj2motor(flat))    # 스파이크/출력
        motor_v = self.motor.v                        # 막전위
        return {
            "goc": goc, "goc_4x4": _goc_4x4,
            "grc": grc, "mli": mli,
            "pkj": pkj, "pkj_2x2": pkj_2x2,
            "motor_y": motor_y,
            "motor_v": motor_v
        }

class CerebellarNet(nn.Module):
    def __init__(self, cfg: Config, action_dim: int):
        super().__init__()
        # Nodes
        self.goc = neuron.LIFNode(tau=cfg.t_goc, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.grc = neuron.LIFNode(tau=cfg.t_grc, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.mli = neuron.LIFNode(tau=cfg.t_mli, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.pkj = neuron.LIFNode(tau=cfg.t_pkj, v_threshold=cfg.th_pkj, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.motor = NonSpikingLIFNode(tau=cfg.t_motor)

        # Convs
        self.mf2goc  = nn.Conv2d(1,        cfg.n_goc, kernel_size=3, stride=1, padding=0, bias=False)  # 10->8
        self.mf2grc  = nn.Conv2d(1,        cfg.n_grc, kernel_size=5, stride=1, padding=0, bias=False)  # 10->6
        self.goc2grc = nn.Conv2d(cfg.n_goc, cfg.n_grc, kernel_size=3, stride=1, padding=0, bias=False) # 8->6
        self.grc2mli = nn.Conv2d(cfg.n_grc, cfg.n_mli, kernel_size=3, stride=1, padding=0, bias=False) # 6->4
        self.grc2pkj = nn.Conv2d(cfg.n_grc, cfg.n_pkj, kernel_size=3, stride=1, padding=0, bias=False) # 6->4
        self.mli2pkj = nn.Conv2d(cfg.n_mli, cfg.n_pkj, kernel_size=1, stride=1, padding=0, bias=False) # 4->4
        #self.cf2pkj  = nn.Linear(cfg.n_cf, cfg.n_pkj * 4 * 4, bias=False)

        # Heads — pkj pooled to 2x2
        self.pkj2motor = nn.Linear(cfg.n_pkj * 2 * 2, action_dim, bias=False)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.1, 0.01)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 1.0, 0.05)
        # Heads — pkj pooled to 2x2
        self.pkj2motor = nn.Linear(cfg.n_pkj * 2 * 2, action_dim, bias=False)

        # init (권장: 평균 0으로)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)

    def forward(self, mf: torch.Tensor) -> Dict[str, torch.Tensor]:
        # goc branch
        goc = self.goc(self.mf2goc(mf))                      # (B, n_goc, 8, 8)
        goc_4x4 = F.avg_pool2d(goc, 2, 2)                    # (B, n_goc, 4, 4)

        # grc/mli
        grc = self.grc(self.mf2grc(mf) - self.goc2grc(goc))  # (B, n_grc, 6, 6)
        mli = self.mli(self.grc2mli(grc))                    # (B, n_mli, 4, 4)

        # pkj maps @ 4x4
        pkj_pre = self.grc2pkj(grc) - self.mli2pkj(mli)      # (B, n_pkj, 4, 4)
        pkj = self.pkj(pkj_pre)                               # (B, n_pkj, 4, 4)
        pkj_2x2 = F.avg_pool2d(pkj, 2, 2)                     # (B, n_pkj, 2, 2)

        # !!! 평탄화는 배치 보존 !!!
        flat = pkj_2x2.flatten(start_dim=1).contiguous()      # (B, n_pkj*2*2)

        # motor head
        motor_y = self.motor(self.pkj2motor(flat))            # (B, action_dim)
        motor_v = self.motor.v                                # (B, action_dim)

        return {
            "goc": goc, "goc_4x4": goc_4x4,
            "grc": grc, "mli": mli,
            "pkj": pkj, "pkj_2x2": pkj_2x2,
            "flat": flat,                     # ← R-STDP용으로 같이 넘겨주면 편함
            "motor_y": motor_y,
            "motor_v": motor_v
        }

# -----------------------------
# Reward-modulated STDP on pkj->motor (Linear)
# -----------------------------
class RewardModulatedSTDP:
    """Implements simple R-STDP on a Linear layer (pkj->motor) using pre/post spike traces.

    Δw ∝  r * [ A_plus * post_spike * pre_trace  -  A_minus * pre_spike * post_trace ]

    - pre spikes: pooled PKJ spikes at 2x2, flattened -> (B, N_pre)
    - post spikes: motor spikes (thresholded from motor LIF output) -> (B, N_post)
    - traces: exponentially decaying with tau_pre/tau_post
    """
    def __init__(self, layer: nn.Linear, n_pre: int, n_post: int, cfg: Config):
        self.layer = layer
        self.n_pre = n_pre
        self.n_post = n_post
        self.A_plus = cfg.A_plus
        self.A_minus = cfg.A_minus
        self.tau_pre = cfg.tau_pre
        self.tau_post = cfg.tau_post
        self.wmin = cfg.wmin
        self.wmax = cfg.wmax
        self.reward_scale = cfg.reward_scale

        # traces
        self.pre_trace = torch.zeros(n_pre, device=layer.weight.device)
        self.post_trace = torch.zeros(n_post, device=layer.weight.device)

        # reward baseline
        self.r_bar = 0.0
        self.ema_beta = cfg.ema_reward_beta

    @torch.no_grad()
    def step(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float):
        # pre_spikes: (B, N_pre) 또는 (B, H, W) -> (B, N_pre)로
        if pre_spikes.dim() == 3:
            B = pre_spikes.size(0)
            pre_spikes = pre_spikes.reshape(B, -1)   # (B, N_pre)

        if post_spikes.dim() == 1:
            post_spikes = post_spikes.unsqueeze(0)   # (1, N_post)

        # --- 배치 평균: (N_pre,), (N_post,) ---
        pre_mean  = pre_spikes.float().mean(dim=0)    # ✅ (N_pre,)
        post_mean = post_spikes.float().mean(dim=0)   # ✅ (N_post,)

        # --- trace decay ---
        a_pre  = math.exp(-1.0 / float(self.tau_pre))
        a_post = math.exp(-1.0 / float(self.tau_post))
        self.pre_trace  = self.pre_trace * a_pre  + pre_mean
        self.post_trace = self.post_trace * a_post + post_mean

        # --- 보상 중심화 ---
        self.r_bar = self.ema_beta * self.r_bar + (1 - self.ema_beta) * float(reward)
        r_hat = float(reward) - self.r_bar

        # --- 가중치 업데이트 ---
        # ltp: post_mean ⊗ pre_trace   (N_post, N_pre)
        # ltd: post_trace ⊗ pre_mean   (N_post, N_pre)
        ltp = torch.outer(post_mean, self.pre_trace)   # ✅ 1D × 1D
        ltd = torch.outer(self.post_trace, pre_mean)   # ✅ 1D × 1D

        dw = self.reward_scale * r_hat * (self.A_plus * ltp - self.A_minus * ltd)

        # in-place update + clamp
        self.layer.weight.add_(dw.to(self.layer.weight.dtype, self.layer.weight.device))
        self.layer.weight.clamp_(self.wmin, self.wmax)

class RewardModulatedSTDP_new:
    """
    R-STDP for a Linear layer (pkj->motor).

    Δw ∝ r̂ * [ A_plus * (post_mean ⊗ pre_trace)  -  A_minus * (post_trace ⊗ pre_mean) ]
    where r̂ = reward - EMA(reward)
    """
    def __init__(self, layer: nn.Linear, n_pre: int, n_post: int, cfg: Config):
        self.layer = layer
        # Enforce consistency with the linear layer
        assert layer.in_features == n_pre, f"in_features({layer.in_features}) != n_pre({n_pre})"
        assert layer.out_features == n_post, f"out_features({layer.out_features}) != n_post({n_post})"

        self.n_pre = n_pre
        self.n_post = n_post
        self.A_plus = cfg.A_plus
        self.A_minus = cfg.A_minus
        self.tau_pre = cfg.tau_pre
        self.tau_post = cfg.tau_post
        self.wmin = cfg.wmin
        self.wmax = cfg.wmax
        self.reward_scale = cfg.reward_scale
        self.ema_beta = cfg.ema_reward_beta

        device = layer.weight.device
        dtype = layer.weight.dtype

        # traces (1D)
        self.pre_trace = torch.zeros(self.n_pre, device=device, dtype=dtype)
        self.post_trace = torch.zeros(self.n_post, device=device, dtype=dtype)

        # reward baseline
        self.r_bar = 0.0

    @torch.no_grad()
    def step(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, reward: float):
        """
        pre_spikes:  (B, N_pre) or (B, H, W)  -> will be flattened to (B, N_pre)
        post_spikes: (B, N_post)
        reward:      float
        """
        device = self.layer.weight.device
        dtype = self.layer.weight.dtype

        # ---- shape sanitize ----
        if pre_spikes.dim() == 3:
            B = pre_spikes.shape[0]
            pre_spikes = pre_spikes.reshape(B, -1)  # flatten to (B, N_pre)
        elif pre_spikes.dim() == 2:
            pass
        else:
            raise ValueError(f"pre_spikes must be (B,N) or (B,H,W), got {pre_spikes.shape}")

        if post_spikes.dim() == 1:
            post_spikes = post_spikes.unsqueeze(0)  # (1, N_post)
        elif post_spikes.dim() != 2:
            raise ValueError(f"post_spikes must be (B,N) or (N,), got {post_spikes.shape}")

        # ---- dimension checks vs layer ----
        if pre_spikes.shape[1] != self.n_pre:
            raise ValueError(f"N_pre mismatch: got {pre_spikes.shape[1]} from pre_spikes, "
                             f"expected {self.n_pre} (= layer.in_features). "
                             f"Flattening or Linear.in_features is wrong.")
        if post_spikes.shape[1] != self.n_post:
            raise ValueError(f"N_post mismatch: got {post_spikes.shape[1]} from post_spikes, "
                             f"expected {self.n_post} (= layer.out_features).")

        # ---- batch means (1D) ----
        pre_mean = pre_spikes.to(dtype).mean(dim=0)   # (N_pre,)
        post_mean = post_spikes.to(dtype).mean(dim=0) # (N_post,)

        # ---- decay traces ----
        alpha_pre  = math.exp(-1.0 / float(self.tau_pre))
        alpha_post = math.exp(-1.0 / float(self.tau_post))
        self.pre_trace.mul_(alpha_pre).add_(pre_mean)
        self.post_trace.mul_(alpha_post).add_(post_mean)

        # ---- reward baseline & centered reward ----
        self.r_bar = self.ema_beta * self.r_bar + (1.0 - self.ema_beta) * float(reward)
        r_hat = float(reward) - self.r_bar

        # ---- weight update ----
        # ltp: (N_post, N_pre) = post_mean ⊗ pre_trace
        # ltd: (N_post, N_pre) = post_trace ⊗ pre_mean
        ltp = torch.outer(post_mean, self.pre_trace)  # (N_post, N_pre)
        ltd = torch.outer(self.post_trace, pre_mean)  # (N_post, N_pre)

        dw = self.reward_scale * r_hat * (self.A_plus * ltp - self.A_minus * ltd)
        self.layer.weight.add_(dw.to(device=device, dtype=dtype))
        self.layer.weight.clamp_(self.wmin, self.wmax)


# -----------------------------
# Training loop (Gymnasium CartPole-v1 demo)
# -----------------------------
class Agent:
    def __init__(self, cfg: Config, env: gym.Env):
        obs_dim = int(np.prod(env.observation_space.shape))
        #n_act = env.action_space.n
        cfg.obs_dim = obs_dim
        cfg.n_motor = 4

        self.cfg = cfg
        self.device = cfg.device
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        #self.encoder = ObsEncoder(obs_dim, cfg.n_cf).to(self.device)
        self.net = CerebellarNet(cfg, action_dim=4).to(self.device)

        # build R-STDP helper for pkj->motor
        n_pre = cfg.n_pkj * 2 * 2
        n_post = 4
        self.rstdp = RewardModulatedSTDP(self.net.pkj2motor, n_pre, n_post, cfg)

    def select_action_old(self, motor_mem_v: torch.Tensor, eps: float) -> torch.Tensor:
        # motor_mem_v: membrane potential vector (B, n_act)
        B = 1
        A = motor_mem_v.shape[0]
        actions = torch.zeros(B, dtype=torch.long, device=motor_mem_v.device)
        for b in range(B):
            if random.random() < eps:
                actions[b] = random.randrange(A)
            else:
                actions[b] = int(torch.argmax(motor_mem_v[b]))
        return actions
    def select_action(self, motor_mem: torch.Tensor, eps: float):
        B = 1
        A = motor_mem.shape[0]
        a = torch.empty(B, dtype=torch.long, device=motor_mem.device)
        for b in range(B):
            if torch.rand(1).item() < eps:
                a[b] = torch.randint(A, (1,)).item()
            else:
                a[b] = torch.argmax(motor_mem[b]).item()
        return a


    def step_episode(self, env):
        obs, _ = env.reset(seed=self.cfg.seed)
        total_reward = 0.0

        for t in range(self.cfg.max_steps):
            mf = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            # 1) encode & forward
            #mf, cf = self.encoder(x)
            functional.reset_net(self.net)  # 매 스텝 상태 초기화 (네 설계 기준)
            outs = self.net(mf)

            # 2) pre/post 추출
            pre = (outs["pkj_2x2"].detach() > 0).float().flatten(1)     # (B, N_pre)
            motor_mem = self.net.motor.v                                 # (B, n_act)
            post = (motor_mem > 0.0).float()                             # (B, n_post)

            # 3) 행동 선택 (막전위 기반 ε-greedy)
            act = int(self.select_action(motor_mem, self.cfg.eps_greedy).item())

            # 4) 환경 진행 → reward 수신
            next_obs, reward, terminated, truncated, info = env.step(act)
            total_reward += reward

            # 5) R-STDP 업데이트 (이제 reward가 정의됨)
            self.rstdp.step(pre, post, float(reward))

            # 6) 종료/이동
            obs = next_obs
            if terminated or truncated:
                break

        return total_reward


def train_plateballenv():
    cfg = Config()
    simcfg = SimulationConfig()
    env = Environment(simcfg)

    agent = Agent(cfg, env)
    rewards = []
    best = -1e9
    for ep in range(cfg.episodes):
        R = agent.step_episode(env)
        rewards.append(R)
        best = max(best, R)
        print(f"Episode {ep+1}/{cfg.episodes}  Reward={R:.1f}  Best={best:.1f}  Baseline={agent.rstdp.r_bar:.3f}")
    env.close()
    return rewards


if __name__ == "__main__":
    train_plateballenv()
