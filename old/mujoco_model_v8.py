#@title Imports and global settings
DEBUG           = True
DEBUG_XML       = False
DEBUG_INIT      = DEBUG and True
DEBUG_RESET     = DEBUG and True
DEBUG_SENSOR    = DEBUG and False
DEBUG_DELAY     = DEBUG and False
DEBUG_STEP      = DEBUG and False



from config import SimulationConfig
import math
import mujoco as mj
import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces  # 파일 상단에 이미 gymnasium import했다면 재사용 가능

import torch
from spikingjelly.activation_based import layer, neuron, functional, learning, surrogate, encoding
from utils_viewer import LiveViewer
# from math import ceil

"""
Major Changes
0. init steps
1. reward is time that arrives at center + time that moves
2. motor moves more slowly
3. ball density increased
4. WTA.
5. Observer
6. Model now has z-axis
7. input monitor is the circuit output
8. motor hinge movement velocity decreased again(1.0->2.0)
9. damping is 1 again
>>>NEW: function to choose motor accumulation or direct control
"""


class DelayLine:
    def __init__(self, max_steps, n_channels):
        self.buf = np.zeros((max_steps+1, n_channels), dtype=np.float32)
        self.ptr = 0
        self.max_steps = max_steps
        self.n = n_channels
    def push(self, x_t):  # x_t shape [n_channels]
        self.buf[self.ptr, :] = x_t
        self.ptr = (self.ptr + 1) % (self.max_steps+1)
    def read_delayed(self, ks):  # ks shape [n_channels], per-channel delay steps
        # gather diagonal indices with wrap-around
        idx = (self.ptr - 1 - ks) % (self.max_steps+1)
        return self.buf[idx, np.arange(self.n)]

class Environment(Env):
    def __init__(self, cfg: SimulationConfig, render = True):
        # config
        self.cfg = cfg
        self.render = render
        self.v = None
        self.plate_size = cfg.plate_size
        self.n_mf = cfg.n_mf
        self.n_motor = cfg.n_motor
        self.dt = cfg.dt  # 초 단위
        # ball params
        self.ball_mass = cfg.ball_mass #(kg)
        self.ball_radius = ((3.0 * self.ball_mass) / (4.0 * math.pi * cfg.ball_density)) ** (1.0 / 3.0) if self.ball_mass else None #(m)
        self.ball_mass_n : float = (self.ball_mass - 0.001) / 0.029 if self.ball_mass else None  # [0,1]로 정규화
        self.ball_density = cfg.ball_density
        #sensor params
        self.max_firing_rate = cfg.max_firing_rate  # Hz

        # Dynamic params
        self.step_count : int = 0
        self.ball_x : float
        self.ball_y : float
        self.success_count : int = 0
        # Simulation params
        self.simulation_duration_timestep : float = cfg.simulation_duration_s/self.dt
        self.success_timestep: int = cfg.success_duration_s/self.dt
        self.success_reward : float = cfg.success_reward_per_sec*self.dt  # 성공시 보너스 (프레임당)
        self.failure_penalty : float = cfg.penalize_failure_per_sec*cfg.dt # 실패시 패널티 (프레임당)
        #self.seed = cfg.seed

        # local params
        self.model: mj.MjModel
        self.data: mj.MjData
        self.sensor_pos_grid = np.linspace(-self.plate_size*0.5*(1-1/self.cfg.n_sensor_1d), self.plate_size*0.5*(1-1/self.cfg.n_sensor_1d), self.cfg.n_sensor_1d)
        self._poisson_encoder = None
        self.gx, self.gy = np.meshgrid(self.sensor_pos_grid, self.sensor_pos_grid, indexing='ij')  # (100,100)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.cfg.n_sensor_1d, self.cfg.n_sensor_1d), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_motor,), dtype=np.float32  # 예: (4,)
        )

        v = max(float(cfg.transmitt_speed), 1e-6)  # [m/s], 0 방지
        dist = np.sqrt(self.gx**2 + self.gy**2)  # (100,100) [m]
        delay_s = dist / v                          # (100,100) [s]
        ks_2d = np.rint(delay_s / self.dt).astype(np.int32)  # (100,100) [steps]
        if DEBUG_INIT: print(ks_2d)

        # DelayLine: 최대 지연 스텝(+1 버퍼)과 채널 수(=n_mf=1000)
        self._delay_ks = ks_2d.reshape(-1)                    # (100,)
        self._delayline = DelayLine(max_steps=int(self._delay_ks.max()), n_channels=self.n_mf)

        # Motor Params
        self.motor_max_radians = math.radians(cfg.motor_max_hinge_deg)
        self.motor_gain = cfg.motor_gain
        self.n_motor = cfg.n_motor
        self.window_len = cfg.motor_window_len  # 창 길이(스텝)
        
        self._ring = np.zeros((self.window_len, self.n_motor), dtype=np.float32)
        self._ring_ptr = 0
        self._ring_sum = np.zeros(self.n_motor, dtype=np.float32)
        '''
        self.window_len = max(1, int(self.motor_acc / self.dt))  # 창 길이(스텝)
        # --- 슬라이딩 평균용 원형버퍼 (합계 O(1) 업데이트) ---
        # --- EMA 상태 ---
        self.use_ema = cfg.use_ema
        self.ema_alpha = cfg.ema_alpha
        self._ema_y = np.zeros(self.n_motor, dtype=np.float32)
        '''
        # ... (환경 초기화 코드)
        if DEBUG_INIT:
            print(f"""
                dt                              {self.dt}
                simulation_duration_timestep    {self.simulation_duration_timestep}
                success_timestep                {self.success_timestep}
                success_reward                  {self.success_reward}
                failure_penalty                 {self.failure_penalty}

                """)
    
    def return_xml(self):
        xml = f"""
        <mujoco model="tilt_plate">
        <compiler angle="degree" inertiafromgeom="true"/>
        <option timestep="{self.dt:.7f}" gravity="0 0 -9.81" integrator="implicit"/>
        <default>
            <geom condim="6" margin="0.0001" solimp="0.9 0.99 0.001" solref="0.005 1.0"/>
            <default class="plate">
                <geom type="box" friction="0.5 0.006 0.0010" rgba="0.8 0.8 0.85 1"/>
            </default>
            <default class="ball">
                <geom type="sphere" friction="0.5 0.006 0.0010" rgba="0.9 0.3 0.3 1"/>
            </default>
            <joint armature="0.002" damping="0.1" limited="true"/>
            <motor gear="1.0" ctrllimited="true" ctrlrange="-1.0 1.0"/>
        </default>
        <worldbody>
            <body name="plate_base" pos="0 0 0">
            <joint name="hinge_x" type="hinge" axis="1 0 0" damping="2.0" range="{-self.cfg.motor_max_hinge_deg} {self.cfg.motor_max_hinge_deg}"/>
            <joint name="hinge_y" type="hinge" axis="0 1 0" damping="2.0" range="{-self.cfg.motor_max_hinge_deg} {self.cfg.motor_max_hinge_deg}"/>
            <geom name="plate_geom" class="plate" size="{self.plate_size/2} {self.plate_size/2} 0.005" mass="1.0"/>
            </body>
            <body name="ball" pos="{self.ball_x} {self.ball_y} {self.ball_radius+0.005:.4f}">
            <freejoint name="ball_free"/>
            <geom name="ball_geom" class="ball" size="{self.ball_radius}" mass="{self.ball_mass}"/>
            </body>
        </worldbody>
        <actuator>
            <!-- ctrl가 '목표 각도[rad]'가 됨. kp↑, kv↑로 강한 서보 구성 -->
            <position name="px" joint="hinge_x" kp="200" kv="10" ctrlrange="{-self.motor_max_radians} {self.motor_max_radians}"/> 
            <position name="py" joint="hinge_y" kp="200" kv="10" ctrlrange="{-self.motor_max_radians} {self.motor_max_radians}"/>
        </actuator>

        </mujoco>
        """.strip()
        xml = f"""<mujoco model="tilt_plate">
        <compiler angle="degree" inertiafromgeom="true"/>
        <option timestep="{self.dt:.7f}" gravity="0 0 -9.81" integrator="implicit"/>

        <default>
            <geom condim="6" margin="0.0001" solimp="0.9 0.99 0.001" solref="0.005 1.0"/>
            <default class="plate">
            <geom type="box" friction="0.5 0.006 0.0010" rgba="0.8 0.8 0.85 1"/>
            </default>
            <default class="ball">
            <geom type="sphere" friction="0.5 0.006 0.0010" rgba="0.9 0.3 0.3 1"/>
            </default>
            <joint armature="0.002" damping="0.1" limited="true"/>
            <motor gear="1.0" ctrllimited="true" ctrlrange="-1.0 1.0"/>
        </default>

        <worldbody>
            <!-- 판 루트: Z 슬라이드 -> X 힌지 -> Y 힌지 -->
            <body name="plate_root" pos="0 0 0">
            <!-- 가운데(중심) 높이: 위/아래 직선 이동 -->
            <joint name="lift_z" type="slide" axis="0 0 1"
                    range="-0.05 0.05" damping="1.0"/>

            <!-- 판 기울기: 두 각도 -->
            <joint name="hinge_x" type="hinge" axis="1 0 0"
                    damping="1.0"
                    range="-{self.cfg.motor_max_hinge_deg} {self.cfg.motor_max_hinge_deg}"/>
            <joint name="hinge_y" type="hinge" axis="0 1 0"
                    damping="1.0"
                    range="-{self.cfg.motor_max_hinge_deg} {self.cfg.motor_max_hinge_deg}"/>

            <geom name="plate_geom" class="plate"
                    size="{self.plate_size/2} {self.plate_size/2} 0.005" mass="1.0"/>
            </body>

            <!-- 공: 초기 z는 (lift_z의 qpos) + 판 반두께(0.005) + 반지름 + 여유 -->
            <body name="ball" pos="{self.ball_x} {self.ball_y} {self.ball_radius+0.005:.4f}">
            <freejoint name="ball_free"/>
            <geom name="ball_geom" class="ball"
                    size="{self.ball_radius}" mass="{self.ball_mass}"/>
            </body>
        </worldbody>

        <actuator>
            <!-- 기존 각도 서보(목표 각도[rad]) -->
            <position name="px" joint="hinge_x" kp="200" kv="10"
                    ctrlrange="-{self.motor_max_radians} {self.motor_max_radians}"/>
            <position name="py" joint="hinge_y" kp="200" kv="10"
                    ctrlrange="-{self.motor_max_radians} {self.motor_max_radians}"/>

            <!-- 새 가운데(중심 높이) 서보(목표 높이[m]) -->
            <position name="pz" joint="lift_z" kp="500" kv="20"
                    ctrlrange="-0.04 0.04"/>
        </actuator>
        </mujoco>
        """.strip()
        if DEBUG_XML: 
            print("xml")
            print(xml)

        return xml
    def close_viewer(self):
        if self.v is not None:
            self.v.close()
            self.v = None
    def reset(self, max_t_change = None, seed=None, options=None):#cfg: SimulationConfig, 
        """
        INPUTS
            seed: 난수 시드 (int) (미사용)
            options: (미사용)
        OUTPUTS
            observation: 초기 센서 입력 (100,100) 또는 (n_mf,) (옵션에 따라 다름)

        Reset ball mass, position, plate angles, and set xml
        """
        # print("Resetting environment...")
        # initialize all state variables
        self.simulation_duration_timestep : float = self.cfg.simulation_duration_s/self.dt
        self.success_timestep: int = self.cfg.success_duration_s/self.dt
        if DEBUG_RESET: print(f"[RESET] sim dur {self.simulation_duration_timestep}, success dur {self.success_timestep}")
        self.step_count = 0
        self.failure_count = 0
        self.total_reward = 0
        self.success_count : int = 0
        if self.cfg.ball_mass is None:
            self.ball_mass = np.random.uniform(0.001, 0.03) #kg
            self.ball_radius = ((3.0 * self.ball_mass) / (4.0 * math.pi * self.ball_density)) ** (1.0 / 3.0) if self.ball_mass else None #(m)
            self.ball_mass_n : float = (self.ball_mass - 0.001) / 0.029 if self.ball_mass else None  # [0,1]로 정규화

        self.ball_x = np.random.uniform(-self.plate_size*0.5+self.ball_radius, self.plate_size*0.5-self.ball_radius)
        self.ball_y = np.random.uniform(-self.plate_size*0.5+self.ball_radius, self.plate_size*0.5-self.ball_radius)
        self._ring = np.zeros((self.window_len, self.n_motor), dtype=np.float32)
        self._ring_ptr = 0
        self._ring_sum = np.zeros(self.n_motor, dtype=np.float32)
        self._delayline = DelayLine(max_steps=int(self._delay_ks.max()), n_channels=self.n_mf)
        # (원한다면 초깃값을 한두 스텝 push해도 됨)


        print(f"ball mass={self.ball_mass*1000:.4f}g (norm = {self.ball_mass_n:.3f}), radius={self.ball_radius*100:.4f}cm ball pos=({self.ball_x*100:.4f}, {self.ball_y*100:.4f})cm")
        

        xml = self.return_xml()
        self.model = mj.MjModel.from_xml_string(xml)
        self.data  = mj.MjData(self.model)
        if self.render: self.v = LiveViewer(self.model, self.data)
        '''
        for _ in range(int(1/self.dt)):
            self.data.ctrl[0] = 0.0
            self.data.ctrl[1] = 0.0
            mj.mj_step(self.model, self.data) 
        # xml에서 정의한 dt만큼 단일 스텝 진행'''

        
        obs = self.sensor_inputs(flatten=False).numpy().astype(np.float32)  # (100,100)
        
        print("[ENV] Initialization Complete", self.data.body("ball").xpos[0:2])
        env_info = {
            "ball_pos": np.array([self.ball_x, self.ball_y], dtype=np.float32),
            "ball_mass": float(self.ball_mass)
        }
        if self.render: self.v.sync()
        return obs, env_info
    
    def _smooth_spikes(self, spikes_motor: np.ndarray) -> np.ndarray:
        """
        spikes_motor: shape (n_motor,), 0 or 1
        return: smoothed (n_motor,), [0,1] 근사율
        """
        x = np.asarray(spikes_motor, dtype=np.float32)
        # 슬라이딩 평균(원형버퍼): 합계 갱신 O(1)
        old = self._ring[self._ring_ptr, :]
        self._ring_sum -= old
        self._ring[self._ring_ptr, :] = x
        self._ring_sum += x
        self._ring_avg = self._ring_sum/self.window_len
        self._ring_ptr = (self._ring_ptr + 1) % self.window_len

        return self._ring_avg

    def _motor_map_to_action(self, motor_sum: np.ndarray) -> np.ndarray:
       
        '''
        sum: (n_motor,) ∈ [0,1]
        - 네가 정의한 4채널 [XP, XN, YP, YN] → position actuator 2채널(px,py)로 합성하거나,
          이미 position actuator 2채널을 쓰면 그 값으로 사용.
        여기서는 두 가지 예시를 제공.'''
        if self.n_motor == 4:
            # 차동 → 두 축 목표각(라디안). ctrlrange가 ±0.087(=±5도)이므로 그 안으로 스케일.
            # 필요시 gain을 곱해 민감도 조절
            ctrl_x = self.motor_gain * (motor_sum[0] - motor_sum[1])    # [-0.087, 0.087] -> 0.2
            ctrl_y = self.motor_gain * (motor_sum[2] - motor_sum[3])
            clipped = np.clip([ctrl_x, ctrl_y], -self.motor_max_radians, self.motor_max_radians)
            return np.array([ctrl_x, ctrl_y], dtype=np.float32)
        elif self.n_motor == 2:
            return motor_sum * self.motor_max_radians
        else:
            # 기타 구조는 사용자가 정의
            raise ValueError(f"Unsupported n_motor={self.n_motor}")
        
    def _edge_to_angles_height(self, action_edge) -> np.ndarray:
        """
        4개 가장자리 높이 -> (theta_x, theta_y, h0)
        z*: float (또는 텐서), Lx/Ly: 판 길이(m)
        """
        Lx, Ly = self.cfg.plate_size, self.cfg.plate_size
        #if DEBUG: print(action_edge)
        zL, zR, zF, zB = (action_edge)*self.motor_gain*(-1)
        theta_x = (zF - zB) / Ly
        theta_y = (zL - zR) / Lx
        h0 = (zL + zR + zB + zF) / 4.0
        return theta_x, theta_y, h0
    
    def sensor_inputs(self, flatten=True):
        """
        OUTPUTS
            inputs: (100,100) spike encoded sensor inputs (torch.float32) if flatten=False
                    (100,) 1D if flatten=True
        """
        n = self.cfg.n_sensor_1d
        # --- 벡터화 준비: grid, ball pos를 torch로 통일 (CPU 기준) ---
        ball_x_t = torch.as_tensor(self.ball_x, dtype=torch.float32)
        ball_y_t = torch.as_tensor(self.ball_y, dtype=torch.float32)

        # --- 거리 및 가우시안 감쇠 (완전 벡터화) ---
        dist2 = ((ball_x_t - self.gx) ** 2 + (ball_y_t - self.gy) ** 2) / 0.18
        sigma = self.cfg.sigma  # 표준편차 = 반지름
        f = torch.exp(-dist2 / (2.0 * sigma * sigma))

        # --- 발화율 및 per-step 확률 계산 (클램프 포함) ---
        #r = torch.clamp((self.ball_mass_n * f) * self.max_firing_rate*self.dt, 0.0, self.max_firing_rate*self.dt)
        p = torch.clamp(f*self.ball_mass_n*self.max_firing_rate*self.dt, 0.0, self.max_firing_rate*self.dt)

        # --- Poisson 인코더: 한 번만 생성해 캐시 ---
        if not hasattr(self, "_poisson_encoder") or self._poisson_encoder is None:
            self._poisson_encoder = encoding.PoissonEncoder()
        pe = self._poisson_encoder

        inputs = pe(p)  # (n,n), torch.float32
        if DEBUG_SENSOR: 
            print(f"[ENV] SENSOR DEBUG")
            print(dist2)
            print(f*100)
            #print("firing Rate")
            #print(r)
            print("firing prob(%)")
            print(p*100)
            print(inputs)

        if flatten:
            inputs = inputs.view(-1)           # (100,)
        else:
            inputs = inputs.view(n, n)         # (10,10)
        return inputs
    
    
    def step(self, action: np.ndarray):
        """
        INPUTS
            action: np.ndarray of shape (4,), values in [0,1] 또는 연속적
                [XP, XN, YP, YN] - 각 모터에 대한 제어 신호
        
        OUTPUTS
            observation: 현재 시간 T에 대한 센서 입력 (n_mf,)
            reward: 보상 (float)
            3초 이상 가운데를 유지하면 terminated=True
            판을 벗어나면 truncated=True
            step_info: 환경 정보 반환 {
                ball_pos: 공 위치 (2,)
                ball_vel: 공 속도 (2,)
                plate_angles_rad: 판 경사각 (2,)
                step_count: 현재 스텝 수 (int)
            }
        """
        self.step_count += 1
        
        #assume action = 0 or 1
        # 
        if self.cfg.motor_mode=='spikes':
            motor_acc = self._smooth_spikes(action)
            action_gain = self._edge_to_angles_height(motor_acc)
            if DEBUG_STEP: print(f"[ENV] MOTOR_ACC: {motor_acc}, ACTION_GAIN: {action_gain}")
            self.data.ctrl[0] = float(action_gain[0])  # px
            self.data.ctrl[1] = float(action_gain[1])  # py
        elif self.cfg.motor_mode=='motor_neuron':
            action_gain = self._edge_to_angles_height(action)
            self.data.ctrl = action_gain
            if DEBUG_STEP: print("[ENV] ACTION / ACTION_GAIN:", action, action_gain)
        mj.mj_step(self.model, self.data) # xml에서 정의한 dt만큼 단일 스텝 진행
        if self.render: self.v.sync()
        
        # TODO: 관측, 보상, 종료, 정보 반환
        ball_pos = self.data.body("ball").xpos  # (3,)
        self.ball_x, self.ball_y = ball_pos[0], ball_pos[1]

        # hinge degreees - in radians
        hinge_x_q = self.data.joint("hinge_x").qpos.copy()
        hinge_y_q = self.data.joint("hinge_y").qpos.copy()
        if DEBUG_STEP: print(f"[ENV] HINGE: ({hinge_x_q}, {hinge_y_q}) ")

        # ----- 관측: (현재 센서스파이크) → DelayLine → (지연 적용 관측) -----
        s_now = self.sensor_inputs(flatten=True).numpy().astype(np.float32)   # (100,)
        self._delayline.push(s_now)
        if DEBUG_DELAY and self._delayline.buf.size != 0:
            print(f"[ENV] DELAY_LINE: {self._delayline.buf}")
        obs_delayed = self._delayline.read_delayed(self._delay_ks).astype(np.float32)  # (100,)
        observation = obs_delayed.reshape(self.cfg.n_sensor_1d, self.cfg.n_sensor_1d) #(10,10)
        # GOES INTO SCNN.

        """
        Rewards/Penalties
        1. Distance from center
        2. center staying time
        3. Penalize - sudden z velocity from board
        4. v direction - #TODO
        """
        dist_from_target = (self.ball_x**2 + self.ball_y**2)*10000 #cm
        ball_vel = self.data.body("ball").cvel
        
        # Velocity Rewards
        dist_from_target = (self.ball_x**2 + self.ball_y**2)*10000 #cm
        ball_vel = self.data.body("ball").cvel
        reward = -dist_from_target  # cm
        reward -= abs(ball_vel[2]) * 100 if abs(ball_vel[2]) > 0.05 else 0
        if DEBUG_STEP: print(f"[ENV] BALL Z VELOCITY: {ball_vel[2]}, penalty = {abs(ball_vel[2]) * 100 if abs(ball_vel[2]) > 0.1 else 0}")

        terminated = False        # terminated if stayed in center for success_timestep
        if dist_from_target < 1:
            self.success_count += 1
            reward *= -self.success_reward  # 보너스 (reward 범위: -1~0)

            if self.success_count >= self.success_timestep:
                terminated = True
                reward += (self.simulation_duration_timestep-self.step_count) * self.success_reward  # 보너스
                print(f"[ENV] Success! Stayed in center for {self.success_timestep} steps.")
        else:
            self.success_count = 0

        truncated = False
        # truncated if out of bounds/sim time over
        if self.step_count >= self.simulation_duration_timestep:
            print(f"[ENV] Last position was ({self.ball_x*100:.4f}cm, {self.ball_y*100:.4f}cm)")
            reward += self.failure_penalty
            truncated = True

        if abs(self.ball_x)>0.15 or abs(self.ball_y)>0.15:
            reward -= (self.simulation_duration_timestep-self.step_count) * self.failure_penalty
            truncated = True
        # if self.failure_count

        if DEBUG_STEP:
            print(f"[ENV] INPUTS:{action} TIMESTEP:{self.step_count:4d} DIST_FROM_TARGET:{dist_from_target*100:.2f} REWARD: {reward:.2f}\t")

        step_info = {
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "plate_angles_rad": np.array([hinge_x_q, hinge_y_q], dtype=np.float32),
            "step_count": self.step_count
        }
        self.total_reward += reward

        if terminated: 
            print(f"[ENV] Step Count: {self.step_count}, Total reward: {self.total_reward}")
        if truncated:
            print(f"[ENV] Step Count: {self.step_count}, Total reward: {self.total_reward}")
        
        return observation, reward, terminated, truncated, step_info
    

  