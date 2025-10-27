from config import SimulationConfig
from mujoco_model_v10 import Environment
from utils_logger import SpikePlotter, SpikeHeatmap, PotentialPlotter, PotentialPlotter_vertical
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

import numpy as np
import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli, Normal
from spikingjelly.activation_based import functional, neuron, surrogate, monitor
from datetime import datetime

from snn_model import SpikingNet

"""
entropy to item.
"""

# SETTINGS
DEBUG = True
DEBUG_MONITOR = False
DEBUG_BY_STEP = False
cfg = SimulationConfig()
device = "xpu" if torch.xpu.is_available() else "cpu"
print(device)



class RolloutStorage:
    def init(self):
        self.obs: list
        self.actions: list
        self.log_probs: list
        self.rewards: list
        self.dones: list  # (terminated or truncated)
        self.values: list
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_env(cfg: SimulationConfig) -> Environment:
    env = Environment(cfg)
    return env

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
    batch_size = states.size(0)

    if mini_batch_size <= 0:
        raise ValueError(f"[ERROR] mini_batch_size must be > 0, got {mini_batch_size}")
    
    if batch_size < mini_batch_size:
        print(f"[WARN] Batch size ({batch_size}) smaller than mini_batch_size ({mini_batch_size}). Using full batch.")
        yield states, actions, log_probs, returns, advantages
        return

    # Ensure we only split full-sized mini-batches
    n_full_batches = batch_size // mini_batch_size
    total_split_size = n_full_batches * mini_batch_size

    ids = np.random.permutation(batch_size)
    ids = np.split(ids[:total_split_size], n_full_batches)

    for i in range(len(ids)):
        yield states[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantages[ids[i], :]

def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    

    if states.size(0) == 0:
        print("[WARN] Empty states passed to PPO update. Skipping update.")
        return
    states   = states.to(device, non_blocking=True)
    actions  = actions.to(device, non_blocking=True).float()
    log_probs, returns, advantages = log_probs.to(device), returns.to(device), advantages.to(device)

    for p_epoch in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            functional.reset_net(model)
            dist, value = model(state)

            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action.float())

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer.step()
            
        
        print(f"[TRAIN] ppo_epoch: {p_epoch}, loss: {loss.item():.4f}, actor_loss: {actor_loss.item():.4f}, critic_loss: {critic_loss.item():.4f}, entropy: {entropy.item():.4f}")

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


class NonSpikingLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def reset(self): 
        # 기존: self.v = torch.tensor(0.0)  # 디바이스/그래프 끊기고 새 텐서 생성 -> 누수/성능저하
        if isinstance(self.v, torch.Tensor):
            with torch.no_grad():
                self.v.zero_()


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

class CerebellarCNNAC2(nn.Module):
    def __init__(self, n_grc=8, n_pkj=32, n_motor=4, tau=cfg.tau, T=cfg.T, std = 0.0, log = False):
        super(CerebellarCNNAC2, self).__init__()

        self.n_grc = n_grc*16

        self.critic = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = n_grc, kernel_size= 4, stride = 2, padding = 'valid', padding_mode = 'zeros', bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Flatten(),
            nn.Linear(self.n_grc, n_pkj, bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Linear(n_pkj, 1, bias = False),
            NonSpikingLIFNode(tau = cfg.motor_decay)
        )
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = n_grc, kernel_size= 4, stride = 2, padding = 'valid', padding_mode = 'zeros', bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Flatten(),
            nn.Linear(self.n_grc, n_pkj, bias = False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            nn.Linear(n_pkj, n_motor, bias = False),
            NonSpikingLIFNode(tau = cfg.motor_decay)
        )
        self.log_std = nn.Parameter(torch.ones(1, 4) * std)
        self.log = log
        self.T = T
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init =="kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
            
            if isinstance(m, nn.Conv2d):
                if cfg.weight_init == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif cfg.weight_init =="kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data)
                elif cfg.weight_init == "normal-0":
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=1.0)
            # monitoring setting
            if isinstance(m, neuron.LIFNode):
                m.store_v_seq = False
        
    def set_logging(self, flag: bool):
        self.log = flag
        if self.log == True:
            self.input_monitor = monitor.InputMonitor(net=self.actor, instance = neuron.LIFNode)
            self.spike_monitor = monitor.OutputMonitor(net=self.actor, instance=neuron.LIFNode, )
            self.potential_monitor = monitor.AttributeMonitor(net=self.actor, pre_forward=False, instance=neuron.LIFNode, attribute_name='v')
        else:
            self.input_monitor = None
            self.spike_monitor = None
            self.potential_monitor = None

    def return_potential_monitor(self):
        if self.log==True:
            #if DEBUG: print(f"[TRAIN] spike monitor records: {self.spike_monitor.records}")n
            step_potential = self.potential_monitor['1'][-1].detach().squeeze().cpu().numpy()
            #step_potential = np.array(self.potential_monitor['1'].detach())
            self.potential_monitor.clear_recorded_data()
            if DEBUG_MONITOR: print(f"[TRAIN] step_potential = {step_potential.flatten()[:20]}...")
            return step_potential
        else: return False
    def return_spike_monitor(self):
        if self.log==True:
            #if DEBUG: print(f"[TRAIN] spike monitor records: {self.spike_monitor.records}")
            step_spike = self.spike_monitor['1'][-1].detach().squeeze().cpu().numpy()
            if DEBUG_MONITOR: print(type(step_spike))
            self.spike_monitor.clear_recorded_data()
            if DEBUG_MONITOR: print(f"[TRAIN] step_spike = {step_spike.flatten()[:20]}...")
            return step_spike
        else: print("error")
    def clear_monitor(self):
        if self.log == True:
            self.spike_monitor.clear_recorded_data()
            self.potential_monitor.clear_recorded_data()
            self.input_monitor.clear_recorded_data()
            
        
    def forward(self, x):
        # ... (SNN 순전파 코드)
        functional.reset_net(self)
        for t in range(self.T):
            self.critic(x)
            self.actor(x)
        value = self.critic[-1].v.clone()
        mu = self.actor[-1].v.clone()
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        #if DEBUG: print(f"[TRAIN] actor:{actor}, critic:{critic}, actor potential:{self.actor[-1].v}, critic potential:{self.critic[-1].v},")
        return dist, value

import torch
import torch.nn as nn
import torch.nn.functional as F

class CerebellarNet(nn.Module):
    """
    Pipeline
      goc = Conv3x3(mf)                                      -> (N, n_goc, 8, 8)
      grc = Conv5x5(mf) + Conv3x3(goc)                        -> (N, n_grc, 6, 6)
      mli = Conv3x3(grc)                                      -> (N, n_mli, 4, 4)
      pkj = Conv3x3(grc)->n_pkj  +  FC(cf)->(n_pkj,4,4) + Conv1x1(mli)->n_pkj  -> (N, n_pkj, 4, 4)
      motor = FC( flatten(pkj) ) -> 4
    """
    def __init__(self, cf_dim: int, n_goc=16, n_grc=32, n_mli=16, n_pkj=32):
        super().__init__()
        # goc
        self.goc = nn.Conv2d(1, n_goc, kernel_size=3, stride=1, padding=0, bias=False)         # 10->8

        # grc (두 경로 채널수를 맞춰야 더할 수 있음)
        self.grc_from_mf  = nn.Conv2d(1,     n_grc, kernel_size=5, stride=1, padding=0, bias=False)  # 10->6
        self.grc_from_goc = nn.Conv2d(n_goc, n_grc, kernel_size=3, stride=1, padding=0, bias=False)  # 8->6

        # mli
        self.mli = nn.Conv2d(n_grc, n_mli, kernel_size=3, stride=1, padding=0, bias=False)      # 6->4

        # pkj: 세 항의 채널/공간 크기를 (n_pkj, 4, 4)로 맞춤
        self.pkj_from_grc = nn.Conv2d(n_grc, n_pkj, kernel_size=3, stride=1, padding=0, bias=False)  # 6->4
        self.pkj_from_mli = nn.Conv2d(n_mli, n_pkj, kernel_size=1, stride=1, padding=0, bias=False)  # 4->4
        # cf를 (n_pkj, 4, 4)로 사상해서 더함
        self.cf_to_map = nn.Linear(cf_dim, n_pkj * 4 * 4, bias=False)

        # motor: pkj를 펼쳐서 4차원으로
        self.motor_head = nn.Linear(n_pkj * 4 * 4, 4, bias=True)

        # 가중치 초기화(선택)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1.0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, mf: torch.Tensor, cf: torch.Tensor):
        """
        mf: (N, 1, 10, 10)
        cf: (N, cf_dim)
        returns:
            motor: (N, 4)
            (원하면 중간 텐서들도 함께 반환 가능)
        """
        assert mf.dim() == 4 and mf.shape[-2:] == (10, 10), "mf must be (N,1,10,10)"
        # goc
        goc = self.goc(mf)                                     # (N, n_goc, 8, 8)

        # grc
        grc_mf  = self.grc_from_mf(mf)                         # (N, n_grc, 6, 6)
        grc_goc = self.grc_from_goc(goc)                       # (N, n_grc, 6, 6)
        grc = grc_mf + grc_goc                                 # (N, n_grc, 6, 6)

        # mli
        mli = self.mli(grc)                                    # (N, n_mli, 4, 4)

        # pkj = conv(grc) + fc(cf) + conv(mli)
        pkj_from_grc = self.pkj_from_grc(grc)                  # (N, n_pkj, 4, 4)
        pkj_from_mli = self.pkj_from_mli(mli)                  # (N, n_pkj, 4, 4)

        cf_map = self.cf_to_map(cf).view(cf.shape[0], -1, 4, 4)  # (N, n_pkj, 4, 4)

        pkj = pkj_from_grc + pkj_from_mli + cf_map             # (N, n_pkj, 4, 4)

        # motor[... , 4] = fc pkj[... , n_pkj]
        motor = self.motor_head(pkj.reshape(pkj.shape[0], -1)) # (N, 4)
        return motor, {"goc": goc, "grc": grc, "mli": mli, "pkj": pkj}



"""
TRAINING

"""
""" 
LOGGING
csv
trial, total steps, reward, cfg
"""

if __name__ == "__main__":
    set_seed(12)
    cfg = SimulationConfig()
    model = CerebellarCNNAC2().to(device)
    env = make_env(cfg)

    optimizer = optim.Adam(model.parameters(), lr=cfg.eta)

    # Create directory (and parent folders if needed)
    DIR = "./TRAIN_PPO"
    os.makedirs(DIR, exist_ok=True)

    start_epoch = 0
    with open(f"{DIR}/MANIFEST.csv", "a") as file_manifest:
        print("[TRAIN] initializing from MANIFEST.csv")
    with open(f"{DIR}/MANIFEST.csv", "r") as file_manifest:
        lines = file_manifest.readlines()
        last_trial = lines[-1] if lines else None
        last_trial_stripped = last_trial.strip() if lines else None
        if last_trial_stripped:
            print(f"[TRAIN] Last trial found: {last_trial_stripped}")
            c = input("[TRAIN] Continue on last trial? (y/n): ")
            if c.lower() == 'y':
                trial_num = int(last_trial_stripped.split(",")[0])
                with open(f'{DIR}/{trial_num}_train/LOG.csv', "r") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    start_epoch = int(rows[-1][0]) + 1 if len(rows) > 1 else 0
                #torch.load_state_dict(model.state_dict(), f"{DIR}/{trial_num-1}_train/model_params.pth")
                model.load_state_dict(torch.load(f"{DIR}/{trial_num}_train/model_params_{start_epoch}.pth"))
            elif c.lower() == 'n':
                trial_num = int(last_trial_stripped.split(",")[0]) + 1
            else:
                print(f"[TRAIN] Invalid input. Exiting.")
                exit(1)
        else:
            trial_num = 0
        print(f"\n[TRAIN] Trial: {last_trial_stripped}")
        print(f"[TRAIN] Config: {cfg}")
    dir_path = f"{DIR}/{trial_num}_train"
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(dir_path+"/plots", exist_ok=True)

    storage = RolloutStorage()
    storage.clear()
    best_return = -float('inf')

    max_epochs = 10
    
    # Get current UTC time
    
    

    for epoch in range(start_epoch, start_epoch + max_epochs):
        '''if epoch % 10 == 0:
            plotif = True
        else:'''
        plotif = False
        model.set_logging(plotif)
            
        # decay simulation time
        max_steps = int(cfg.simulation_duration_s / cfg.dt)
        ret = env.reset()
        observation, info = ret
        ball_mass = info["ball_mass"]
        ball_pos = info["ball_pos"]

        print(f"[TRAIN] info:{info}\nball_pos_x: {ball_pos[0]}, ball_pos_y: {ball_pos[1]}, ball_mass: {ball_mass*1000}g")
        if DEBUG: print("[TRAIN] Loading Epoch.")
        if DEBUG:
            print(f"[TRAIN] Initial observation shape: {observation.shape}")
            print(f"[TRAIN] Info: {info}")
        entropy = 0.0
        total_reward = 0.0
        if plotif: 
            logger_spike = SpikePlotter(plot_name=f"{DIR}/{trial_num}_train/plots/sensor_spike_epoch_{epoch}")
            logger_cnn = SpikePlotter(plot_name=f"{DIR}/{trial_num}_train/plots/cnn_spike_epoch_{epoch}")
            logger_cnnpotential = PotentialPlotter(plot_name=f"{DIR}/{trial_num}_train/plots/cnnpotential_spike_epoch_{epoch}", plot_size=(20, 20))
            logger_action = SpikePlotter(plot_name=f"{DIR}/{trial_num}_train/plots/output_spike_epoch_{epoch}")
            logger_heatmap = SpikeHeatmap(plot_name=f"{DIR}/{trial_num}_train/plots/sensor_heatmap_epoch_{epoch}", shape=(cfg.n_sensor_1d, cfg.n_sensor_1d))

        duration = max_steps
        current_step = 0


        if DEBUG: print("[TRAIN] Entering Loop...")
        while True:
            if current_step == 0 and DEBUG: print("[TRAIN] Saving spikes...")
            if plotif: 
                logger_heatmap.save_spikes(observation)
                if DEBUG_BY_STEP: 
                    print("LOG - OBSERVATION:", observation.flatten())
                logger_spike.save_spikes(observation.flatten())
            if current_step == 0 and DEBUG: print("[TRAIN] Sampling action...")
            
            observation = torch.from_numpy(observation).float().to(device)
            observation = observation.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            
            torch.autograd.set_detect_anomaly(True)
            distribution, value = model(observation)
            action = distribution.sample()
            action_tensor = action.detach()
            if DEBUG_BY_STEP: print("[TRAIN] Distribution, Value: ", distribution, value)
            if DEBUG_BY_STEP: print("[TRAIN] action: ",action)   

            log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
            entropy += distribution.entropy().mean().item()
            action_np = action.cpu().numpy()[0]
            #action = [1 if a >= 1 else 0 for a in action]  
            if DEBUG_BY_STEP:
                print("LOG_PROB:", log_prob)

            if DEBUG_BY_STEP:
                print("[TRAIN] obs_shape: ", observation.shape)
                print("[TRAIN] Distribution: ", distribution)
                print("[TRAIN] Value: ", value)
                print("[TRAIN] Log_prob: ", log_prob)
                print("[TRAIN] Entropy: ", entropy)
                print("[TRAIN] Action: ", action)
            
            if DEBUG_MONITOR and DEBUG_BY_STEP:
                print(model.actor)
                print("[TRAIN] Input Monitor")
                print(model.input_monitor['1'][-1].flatten())
                print("[TRAIN] Spike Monitor")
                print(model.spike_monitor['1'][-1].flatten())
                print("[TRAIN] Potential Monitor")
                print(model.potential_monitor['1'][-1].flatten())

            if plotif: 
                logger_action.save_spikes(action)
                cnn_spikes = model.return_spike_monitor()
                logger_cnn.save_spikes(cnn_spikes)
                if DEBUG_BY_STEP: print("CNN_SPIKES:", cnn_spikes.flatten())
                cnn_potentials = model.return_potential_monitor()
                if DEBUG_BY_STEP: print("CNN_POTENTIALS:", cnn_potentials.flatten())
                logger_cnnpotential.save_spikes(cnn_potentials)
                
                model.clear_monitor()
            # Move to next step
                
            if DEBUG_BY_STEP: print("[TRAIN] action: ",action)
            next_observation, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            done = terminated or truncated

            if current_step == 0 and DEBUG:
                print("[TRAIN] Storing data... ")

            storage.log_probs.append(log_prob.detach())
            storage.values.append(value.detach())
            storage.rewards.append(torch.tensor([[reward]], dtype=torch.float32, device=device))
            storage.dones.append(torch.tensor([[1 - (terminated or truncated)]], dtype=torch.float32, device=device))
            storage.obs.append(observation.detach().cpu())        # 관측은 CPU에 저장해 메모리 압박 완화
            storage.actions.append(action_tensor.detach().cpu())
            if DEBUG_BY_STEP: print("ACTION: ", storage.actions)
            observation = next_observation

            current_step += 1
            
            if DEBUG_BY_STEP: 
                temp = input("Continue to next step?")
                if temp.lower() == 'n': break
            if terminated or truncated:
                if DEBUG: print("[TRAIN] Returning to main loop")
                duration = current_step
                break

        if plotif:
            if DEBUG: print("[TRAIN] Plot Spikes...")
                
            logger_spike.plot()
            logger_spike.clear()

            logger_cnn.plot()
            logger_cnn.clear()

            logger_cnnpotential.plot()
            logger_cnnpotential.clear()

            logger_heatmap.plot()
            logger_heatmap.clear()

            logger_action.plot()
            logger_action.clear()
        if isinstance(next_observation, np.generic):
            print("next_observation is np.generic.")
            next_observation = np.array([next_observation])  # Wrap scalar
        if DEBUG: print("[TRAIN] Obtaining next value...")
        
        if DEBUG: print("[TRAIN]    next_observation = torch.tensor(next_observation, dtype=torch.float32).to(device)")
        next_observation = torch.tensor(next_observation, dtype=torch.float32).to(device)
        
        if DEBUG: print("[TRAIN]    next_observation = next_observation.unsqueeze(0).unsqueeze(0)")
        next_observation = next_observation.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        if DEBUG: print("[TRAIN]    _, next_value = model(next_observation)")

        functional.reset_net(model)
        with torch.no_grad():
            _, next_value = model(next_observation)
        if DEBUG: print("[TRAIN] Resetting Model...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if DEBUG: print("[TRAIN] Computing GAE...")
        returns = compute_gae(next_value, storage.rewards, storage.dones, storage.values)

        if DEBUG: print("[TRAIN] Computing PPO update...")
        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(storage.log_probs).detach()
        values    = torch.cat(storage.values).detach()
        states    = torch.cat(storage.obs)
        actions   = torch.cat(storage.actions)
        advantage = returns - values

        ppo_epochs = 20
        mini_batch_size = 64
        if DEBUG: print(f"[TRAIN] Actions: {actions[0]} ...")
        ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
        functional.reset_net(model)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"): torch.xpu.empty_cache()
        storage.clear()
        
        now_utc = datetime.now()
        # Extract components
        month = now_utc.month
        day = now_utc.day
        hour = now_utc.hour
        minute = now_utc.minute
        second = now_utc.second
        print(f"[TRAIN] [Epoch {epoch}] total_reward: {total_reward:.2f}, duration: {duration}, entropy: {entropy:.4f}, datetime:{month:02}/{day:02} {hour:02}:{minute:02}:{second:02}")
        with open(f"{DIR}/{trial_num}_train/LOG.csv", "a") as file_train_log:
            file_train_log.write(f"{epoch},{month:02}/{day:02} {hour:02}:{minute:02}:{second:02},{duration},{total_reward},{ball_pos[0]},{ball_pos[1]},{ball_mass*1000}\n")
        storage.clear()
        env.close_viewer()
            
    if DEBUG: print("[TRAIN] Training complete.")
    torch.save(model.state_dict(), f"{DIR}/{trial_num}_train/model_params_{start_epoch+max_epochs}.pth")
    print(model.state_dict())

    with open(f"{DIR}/{trial_num}_train/CONFIG.txt", "w") as file_config:
        file_config.write(str(cfg))

    with open(f"{DIR}/{trial_num}_train/model_params_{start_epoch+max_epochs}.csv", "a") as file_params:
        writer = csv.writer(file_params)
        writer.writerow(["Layer", "Parameter Name", "Values"])
        for name, param in model.state_dict().items():
            #param_flat = param.cpu().numpy().flatten()
            #writer.writerow([name.split('.')[0], name, param_flat.tolist()])
            file_params.write(f"\n{name.split('.')[0]},{name}\n")
            for items in param.cpu().numpy():
                if items.ndim >= 1:
                    for it in items:
                        if it.ndim >= 1:
                            for it2 in it:
                                if it2.ndim >= 1:
                                    for it3 in it2:
                                        file_params.write(f"{it3},")
                                else: 
                                    file_params.write(f"{it2}")
                                file_params.write("\n")
                        else:
                            file_params.write(f"{it}\n")
                else:
                    file_params.write(f"{items}\n")

    with open(f"{DIR}/MANIFEST.csv", "a") as file_manifest:
        file_manifest.write(f"{trial_num},{total_reward},{cfg}\n")
    if DEBUG: print("[TRAIN] All data saved.")

    
    # Evaluation is on another code
