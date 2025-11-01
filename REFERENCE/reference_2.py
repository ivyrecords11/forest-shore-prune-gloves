import os, sys, csv
import numpy as np
from mujoco_model_v11 import Environment
from dqn_5.config_cnn2 import SimulationConfig

cfg = SimulationConfig()
env = Environment(cfg)

DEBUG = True

# ------------------------------------------------------------
# 1. 전역 파라미터/shape
# ------------------------------------------------------------
C_GRC = cfg.c_grc
C_GOC = cfg.c_goc
C_BKC = cfg.c_bkc
C_PKJ = cfg.c_pkj
N_PKJ = C_PKJ * 4     # torch 버전에서 이렇게 했으니까 그대로
N_MOTOR = 4

TAU_GRC = cfg.t_grc
TAU_GOC = cfg.t_goc
TAU_BKC = cfg.t_bkc
TAU_PKJ = cfg.t_pkj
TAU_MOTOR = cfg.motor_decay

THRESHOLD = 1.0

# 얘는 네가 저장한 CSV 위치
WEIGHT_PATH = r"dqn_5\symmetric2\TRAIN\20_train\model_csv\model_params_37_q78.csv"

# ------------------------------------------------------------
# 2. weight 메모리 준비 (torch 모듈과 이름 맞춤)
#   Conv 형식: (out_ch, in_ch, kH, kW)
#   Linear:    (out, in)
# ------------------------------------------------------------
w_mf2goc   = np.zeros((C_GOC, 1,      5, 5), dtype=np.float32)
w_goc2grc  = np.zeros((C_GRC, C_GOC,  2, 2), dtype=np.float32)
w_mf2grc   = np.zeros((C_GRC, 1,      2, 2), dtype=np.float32)  # stride=2지만 weight shape는 2x2
w_pf2bkc   = np.zeros((C_BKC, C_GRC,  3, 3), dtype=np.float32)
w_bkc2pkj  = np.zeros((C_PKJ, C_BKC,  2, 2), dtype=np.float32)
w_pf2pkj   = np.zeros((C_PKJ, C_GRC,  3, 3), dtype=np.float32)
w_pkj2motor= np.zeros((N_MOTOR, N_PKJ),      dtype=np.float32)

# ------------------------------------------------------------
# 3. LIF 노드 (numpy 버전)
# ------------------------------------------------------------
def lif_step(v, I, tau, v_th=1.0, v_reset=0.0):
    v = v + I
    v = v - v / tau
    spike = (v >= v_th).astype(np.float32)
    v = np.where(spike > 0, v_reset, v)
    return v, spike

def nonspiking_lif_step(v, I, tau):
    v = v + I
    v = v - v / tau
    return v

# ------------------------------------------------------------
# 4. 2D conv (valid) + stride 지원
#    x: (C_in, H, W)
#    w: (C_out, C_in, kH, kW)
# ------------------------------------------------------------
def conv2d_valid(x, w, stride=1):
    C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    s = stride
    H_out = (H - kH) // s + 1
    W_out = (W - kW) // s + 1
    y = np.zeros((C_out, H_out, W_out), dtype=np.float32)
    for oc in range(C_out):
        for oh in range(H_out):
            for ow in range(W_out):
                h0 = oh * s
                w0 = ow * s
                patch = x[:, h0:h0+kH, w0:w0+kW]  # (C_in, kH, kW)
                y[oc, oh, ow] = np.sum(patch * w[oc, :, :, :])
    return y

# ------------------------------------------------------------
# 5. CSV 로더 (전에 준 거 구조만 이름 맞춤)
# ------------------------------------------------------------
def _iter_csv_blocks(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline()
        cur_layer, cur_name, cur_vals = None, None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                if cur_layer is not None and cur_name is not None:
                    yield cur_layer, cur_name, cur_vals
                parts = line.split(",", 1)
                cur_layer = parts[0].strip()
                cur_name = parts[1].strip()
                cur_vals = []
            else:
                nums = [x for x in line.split(",") if x != ""]
                cur_vals.extend([float(x) for x in nums])
        if cur_layer is not None and cur_name is not None:
            yield cur_layer, cur_name, cur_vals

def load_weights_from_manifest(filepath: str):
    global w_mf2goc, w_goc2grc, w_mf2grc, w_pf2bkc, w_bkc2pkj, w_pf2pkj, w_pkj2motor

    MAPPING = {
        # CSV param name : (target, shape)
        "base.mf2goc.weight":   ("w_mf2goc",   w_mf2goc.shape),
        "base.goc2grc.weight":  ("w_goc2grc",  w_goc2grc.shape),
        "base.mf2grc.weight":   ("w_mf2grc",   w_mf2grc.shape),
        "base.pf2bkc.weight":   ("w_pf2bkc",   w_pf2bkc.shape),
        "base.bkc2pkj.weight":  ("w_bkc2pkj",  w_bkc2pkj.shape),
        "base.pf2pkj.weight":   ("w_pf2pkj",   w_pf2pkj.shape),
        "base.pkj2motor.weight":("w_pkj2motor",w_pkj2motor.shape),
    }

    loaded = {}
    for layer, name, values in _iter_csv_blocks(filepath):
        key = name
        if key not in MAPPING:
            continue
        target_name, shape = MAPPING[key]
        arr = np.array(values, dtype=np.float32)
        if arr.size != np.prod(shape):
            raise ValueError(f"{key}: need {np.prod(shape)}, got {arr.size}")
        arr = arr.reshape(shape)
        if target_name == "w_mf2goc":
            w_mf2goc[...] = arr
        elif target_name == "w_goc2grc":
            w_goc2grc[...] = arr
        elif target_name == "w_mf2grc":
            w_mf2grc[...] = arr
        elif target_name == "w_pf2bkc":
            w_pf2bkc[...] = arr
        elif target_name == "w_bkc2pkj":
            w_bkc2pkj[...] = arr
        elif target_name == "w_pf2pkj":
            w_pf2pkj[...] = arr
        elif target_name == "w_pkj2motor":
            w_pkj2motor[...] = arr
        loaded[key] = arr
    return loaded

# 실제로 한 번 로드
_ = load_weights_from_manifest(WEIGHT_PATH)
if DEBUG:
    print("[DEBUG] weights loaded")

# ------------------------------------------------------------
# 6. 상태 메모리 준비 (LIF voltages)
# ------------------------------------------------------------
# conv 계층들은 time-step마다 새로 계산하니까 v는 feature map 크기대로 잡아야 하는데
# 여기선 간단히 “직전 time의 출력을 v로 둔다” 식으로 할 거야.
v_goc = None
v_grc = None
v_bkc = None
v_pkj = None
v_motor = np.zeros((N_MOTOR,), dtype=np.float32)

# ------------------------------------------------------------
# 7. 시뮬레이션 루프
# ------------------------------------------------------------
action = np.zeros((4,), dtype=np.float32)
sim_dur_step = int(cfg.simulation_duration_s / cfg.dt)

for step in range(sim_dur_step):
    mf, reward, terminated, truncated, info = env.step(action)
    # mf: (1, 10, 10) 이라고 가정
    mf = np.array(mf, dtype=np.float32).reshape(1, 10, 10)

    # 1) mf -> goc
    goc_I = conv2d_valid(mf, w_mf2goc, stride=1)  # (C_GOC, H1, W1)
    if v_goc is None:
        v_goc = np.zeros_like(goc_I)
    v_goc, s_goc = lif_step(v_goc, goc_I, TAU_GOC, v_th=THRESHOLD)
    goc_out = -s_goc  # torch 코드처럼 음수로 넘김

    # 2) mf -> grc  (stride=2)
    grc_from_mf = conv2d_valid(mf, w_mf2grc, stride=2)      # (C_GRC, H2, W2)
    # goc -> grc
    grc_from_goc = conv2d_valid(goc_out, w_goc2grc, stride=1)  # 크기가 다를 수 있음

    # 두 출력을 더하려면 크기를 맞춰야 함. 여기서는 더 작은 쪽으로 crop
    Hm, Wm = grc_from_mf.shape[1], grc_from_mf.shape[2]
    Hg, Wg = grc_from_goc.shape[1], grc_from_goc.shape[2]
    Hc = min(Hm, Hg)
    Wc = min(Wm, Wg)
    grc_I = grc_from_mf[:, :Hc, :Wc] + grc_from_goc[:, :Hc, :Wc]

    if v_grc is None:
        v_grc = np.zeros_like(grc_I)
    v_grc, s_grc = lif_step(v_grc, grc_I, TAU_GRC, v_th=THRESHOLD)

    # 3) grc -> bkc  (pf2bkc, stride=1)
    bkc_I = conv2d_valid(s_grc, w_pf2bkc, stride=1)
    if v_bkc is None:
        v_bkc = np.zeros_like(bkc_I)
    v_bkc, s_bkc = lif_step(v_bkc, bkc_I, TAU_BKC, v_th=THRESHOLD)
    bkc_out = -s_bkc

    # 4) grc -> pkj  (pf2pkj, stride=2)
    pkj_from_grc = conv2d_valid(s_grc, w_pf2pkj, stride=2)
    #    bkc -> pkj  (bkc2pkj, stride=1)
    pkj_from_bkc = conv2d_valid(bkc_out, w_bkc2pkj, stride=1)

    Hp1, Wp1 = pkj_from_grc.shape[1], pkj_from_grc.shape[2]
    Hp2, Wp2 = pkj_from_bkc.shape[1], pkj_from_bkc.shape[2]
    Hp = min(Hp1, Hp2)
    Wp = min(Wp1, Wp2)

    pkj_I = pkj_from_grc[:, :Hp, :Wp] + pkj_from_bkc[:, :Hp, :Wp]

    if v_pkj is None:
        v_pkj = np.zeros_like(pkj_I)
    v_pkj, s_pkj = lif_step(v_pkj, pkj_I, TAU_PKJ, v_th=THRESHOLD)
    pkj_out = -s_pkj

    # 5) pkj -> motor (linear)
    # pkj_out: (C_PKJ, Hp, Wp)
    flat_pkj = pkj_out.reshape(-1)  # len = C_PKJ * Hp * Wp
    # 여기서 길이가 N_PKJ와 안 맞을 수 있으니 잘라주자
    use_len = min(N_PKJ, flat_pkj.shape[0])
    in_vec = np.zeros((N_PKJ,), dtype=np.float32)
    in_vec[:use_len] = flat_pkj[:use_len]

    motor_I = w_pkj2motor @ in_vec   # (4,)

    # non-spiking LIF
    v_motor = nonspiking_lif_step(v_motor, motor_I, TAU_MOTOR)

    # inhibit mask 적용
    # motor' = motor - A_mask @ motor
    v_motor = v_motor - (A_mask @ v_motor)

    # action 결정 (binary)
    action = (v_motor > 0.0).astype(np.float32)

    if DEBUG and step % 50 == 0:
        print(f"[STEP {step}] motor = {v_motor}, action = {action}")

    if terminated or truncated:
        env.reset()
        v_goc = v_grc = v_bkc = v_pkj = None
        v_motor = np.zeros((N_MOTOR,), dtype=np.float32)
        action = np.zeros((4,), dtype=np.float32)
