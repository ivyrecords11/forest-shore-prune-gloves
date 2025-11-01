import os, time, argparse, csv, math, random, collections, gc
import sys

# 현재 파일 기준으로 상위 폴더 경로 추가
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


from dqn_5.config_cnn2 import SimulationConfig
from mujoco_model_v11 import Environment
from utils_logger import SpikePlotter, SpikeHeatmap

import numpy as np
cfg = SimulationConfig()

DEBUG = True

# TODO: Parameter 불러오기
# TODO: 뉴런 설계
# TODO: 연산기 설계
# TODO: 각 단계 컨트롤러 설계 - 안해도 될듯?
# TODO: 각 단계 save

import csv
import numpy as np

#csv reader helper
def _iter_csv_blocks(csv_path: str):
    """
    CSV 예시:
    Layer,Parameter Name,Values
    base,base.mf2grc.weight
    0.1,0.2,0.3
    0.4
    base,base.mf2goc.weight
    1.0,1.1
    ...
    이런 식으로 돼 있다고 가정.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        # 첫 줄(header) 버림
        header = f.readline()
        cur_layer, cur_name, cur_vals = None, None, []

        for line in f:
            line = line.strip()
            if not line:
                continue

            # 새 파라미터 시작: 콤마가 있는 줄
            if "," in line:
                # 이전 파라미터 flush
                if cur_layer is not None and cur_name is not None:
                    yield cur_layer, cur_name, cur_vals
                parts = line.split(",", 1)
                cur_layer = parts[0].strip()
                cur_name = parts[1].strip()
                cur_vals = []
            else:
                # 숫자 줄
                nums = [x for x in line.split(",") if x != ""]
                cur_vals.extend([float(x) for x in nums])

        # 마지막 파라미터 flush
        if cur_layer is not None and cur_name is not None:
            yield cur_layer, cur_name, cur_vals

def load_weights_from_manifest(filepath: str, cfg):
    """
    filepath: CSV 경로
    cfg: SimulationConfig
    return: dict 형태로 돌려주고, 동시에 전역 w_.. 배열에 채워 넣는다.
    """
    # 전역 배열을 쓰려면 global 선언
    global w_grc, w_goc, w_g2g, w_pkj, w_bkc, w_b2p, w_p2m

    # 네가 가진 csv에서 어떤 이름이 어떤 레이어에 대응되는지 여기서만 정의
    # <<< 이 부분은 네 CSV 실제 이름에 맞게 바꿔라 >>>
    MAPPING = {
        # csv_param_name: ("target", shape_tuple)
        "base.mf2grc.weight": ("w_grc", (1, cfg.c_grc, cfg.f_grc, cfg.f_grc)),
        "base.mf2goc.weight": ("w_goc", (1, cfg.c_goc, cfg.f_goc, cfg.f_goc)),
        "base.goc2grc.weight": ("w_g2g", (cfg.c_goc, cfg.c_grc, cfg.f_goc, cfg.f_goc)),
        "base.grc2pkj.weight": ("w_pkj", (cfg.c_grc, cfg.c_pkj, cfg.f_pkj, cfg.f_pkj)),
        "base.grc2bkc.weight": ("w_bkc", (cfg.c_grc, cfg.c_bkc, cfg.f_b2p, cfg.f_b2p)),
        "base.bkc2pkj.weight": ("w_b2p", (cfg.c_bkc, cfg.c_pkj, cfg.f_b2p, cfg.f_b2p)),
        "base.pkj2motor.weight": ("w_p2m", (cfg.n_pkj, 4)),
    }
    loaded = {}

    for layer, name, values in _iter_csv_blocks(filepath):
        key = name  # ex) "base.mf2grc.weight"
        if key not in MAPPING:
            # 모르는 파라미터면 그냥 건너뛴다
            continue

        target_name, shape = MAPPING[key]
        arr = np.array(values, dtype=np.float32)

        need_elems = 1
        for s in shape:
            need_elems *= s

        if arr.size != need_elems:
            raise ValueError(
                f"[load_weights_from_manifest] {key} needs {need_elems} elems, but CSV has {arr.size}"
            )

        arr = arr.reshape(shape)

        # 전역 배열에다가 복사
        if target_name == "w_grc":
            w_grc[...] = arr
        elif target_name == "w_goc":
            w_goc[...] = arr
        elif target_name == "w_g2g":
            w_g2g[...] = arr
        elif target_name == "w_pkj":
            w_pkj[...] = arr
        elif target_name == "w_bkc":
            w_bkc[...] = arr
        elif target_name == "w_b2p":
            w_b2p[...] = arr
        elif target_name == "w_p2m":
            w_p2m[...] = arr
        else:
            # 여기 안 오게 돼있음
            pass

        loaded[key] = arr

    return loaded

def save_outputs_to_file(filepath: str):
    #TODO
    #open directory as f
    return 0
env = Environment(cfg)
# 초기값 - 로직에서 전부 프로그래머블 해야 함
N_INPUT = 100
BUFFER_SIZING = 'maximum' # or optimal
OPTIMAL_BUFFER_SIZE = 0
THRESHOLD = 1

C_GRC   = cfg.c_grc;    C_GOC   = cfg.c_goc;    C_BKC   = cfg.c_bkc;    C_PKJ   = cfg.c_pkj
N_GRC   = cfg.n_grc;    N_GOC   = cfg.n_goc;    N_BKC   = cfg.n_bkc;    N_PKJ   = cfg.n_pkj;    N_MOTOR = 4
TAU_GRC = cfg.t_grc;    TAU_GOC = cfg.t_goc;    TAU_BKC = cfg.t_bkc;    TAU_PKJ = cfg.t_pkj;    TAU_MOTOR = cfg.motor_decay
F_GRC   = cfg.f_grc;    F_GOC   = cfg.f_goc;    F_G2G   = cfg.f_g2g;    F_BKC   = cfg.f_bkc;    F_PKJ   = cfg.f_pkj;    F_B2P   = cfg.f_b2p

# MEMORY-WEIGHTS( cpre*cpost*filter_w**2)
# load from file
WEIGHT_PATH = "dqn_5/symmetric2/TRAIN/20_train/model_csv/model_params_37.csv"

w_grc = np.zeros((1,       C_GRC, F_GRC, F_GRC), dtype=np.float16)
w_goc = np.zeros((1,       C_GOC, F_GOC, F_GOC), dtype=np.float16)
w_g2g = np.zeros((C_GOC,   C_GRC, F_GOC, F_GOC), dtype=np.float16)
w_pkj = np.zeros((C_GRC,   C_PKJ, F_PKJ, F_PKJ), dtype=np.float16)
w_bkc = np.zeros((C_GRC,   C_BKC, F_B2P, F_B2P), dtype=np.float16)
w_b2p = np.zeros((C_BKC,   C_PKJ, F_B2P, F_B2P), dtype=np.float16)
w_p2m = np.zeros((N_PKJ,   4),                 dtype=np.float16)

# ---- 여기서 실제 로드 ----
loaded_weights = load_weights_from_manifest(WEIGHT_PATH, cfg)
if DEBUG:
    print("[DEBUG] loaded params:", list(loaded_weights.keys()))

# this is also the address obtaining formula (pre channel * post channel * height * width)

# MEMORY - POTENTIAL
v_grc = np.zeros(N_GRC)
v_goc = np.zeros(N_GOC)
v_pkj = np.zeros(N_PKJ)
v_bkc = np.zeros(N_BKC)
v_motor = np.zeros(N_MOTOR)
v_motor_output = np.zeros(N_MOTOR)

# SPIKES - dunno i will use this
s_grc = np.zeros(N_GRC)
s_goc = np.zeros(N_GOC)
s_pkj = np.zeros(N_PKJ)
s_bkc = np.zeros(N_BKC)
s_motor = np.zeros(N_MOTOR)


action = np.array([0,0,0,0])
sim_dur_step = int(cfg.simulation_duration_s/cfg.dt)

for i in range(sim_dur_step):
    # 1) 환경에서 입력 받아오기
    mf, reward, terminated, truncated, info = env.step(action)

    # 2) 입력 전처리 -------------------------------------------------
    # mf를 (10,10)으로 만든다
    mf = np.array(mf, dtype=np.float32).reshape(10, 10)

    # 간단한 큐 (지금은 그냥 자리만 만들어둠)
    spike_queue_mf = []
    queue_ptr_mf = 0
    mf_flat = mf.reshape(-1)  # (100,)

    for spike in mf_flat:
        if spike:       # 0이 아니면 스파이크로 본다
            spike_queue_mf.append(1)
            queue_ptr_mf += 1
        else:
            spike_queue_mf.append(0)

    if DEBUG:
        print(f"[STEP {i}] mf spikes in queue:", queue_ptr_mf)

    # ------------------------------------------------------------
    # 3) mf -> grc : conv 스타일로 계산 (입력: 1채널 10x10, 커널: C_GRC, k=F_GRC)
    #    w_grc shape = (1, C_GRC, F_GRC, F_GRC)
    # ------------------------------------------------------------
    F = F_GRC
    H_in, W_in = 10, 10
    H_out = H_in - F + 1
    W_out = W_in - F + 1

    # 출력 feature map을 잠깐 담아둘 곳
    grc_feature = np.zeros((C_GRC, H_out, W_out), dtype=np.float32)

    for oc in range(C_GRC):
        kernel = w_grc[0, oc, :, :]  # (F, F)
        for oh in range(H_out):
            for ow in range(W_out):
                patch = mf[oh:oh+F, ow:ow+F]  # (F, F)
                grc_feature[oc, oh, ow] = np.sum(patch * kernel)

    if DEBUG and i == 0:
        # 너 예전에 5x5 찍었던 거랑 비슷하게만 보여줌
        print("[DEBUG] GRC feature map (0번 채널) 5x5 일부:")
        view_h = min(5, H_out)
        view_w = min(5, W_out)
        for h in range(view_h):
            for w in range(view_w):
                print(f"{grc_feature[0, h, w]:6.3f}", end=" ")
            print()

    # ------------------------------------------------------------
    # 4) feature -> 실제 granule neuron voltage 로 옮기기
    #    방법: 그냥 다 펴서 N_GRC에 앞에서부터 넣고 남으면 버림
    # ------------------------------------------------------------
    grc_vec = grc_feature.reshape(-1)  # (C_GRC * H_out * W_out,)
    use_len = min(N_GRC, grc_vec.shape[0])

    # LIF 업데이트
    for r in range(use_len):
        # 입력전류 I를 conv 결과로 본다
        I = grc_vec[r]
        v_grc[r] += I
        # leak
        v_grc[r] -= v_grc[r] / TAU_GRC
        if v_grc[r] >= THRESHOLD:
            s_grc[r] = 1
            v_grc[r] = 0.0
        else:
            s_grc[r] = 0

    # 남는 뉴런은 그냥 leak만 시킴
    for r in range(use_len, N_GRC):
        v_grc[r] -= v_grc[r] / TAU_GRC
        s_grc[r] = 0

    # ------------------------------------------------------------
    # 5) grc -> pkj fully-connected (w_pkj shape = (C_GRC, C_PKJ, F_PKJ, F_PKJ))
    #    일단 단순화해서: granule spike 벡터 -> PKJ 전위
    #    여기서는 w_pkj를 그냥 (N_GRC, N_PKJ)로 본다고 치고 합산만 할게
    #    실제 CSV가 conv형이면 위랑 똑같이 풀어서 써라.
    # ------------------------------------------------------------
    # 일단 한 번만 만들게
    if i == 0 and DEBUG:
        print("[DEBUG] GRC spikes (first 16):", s_grc[:16])

    # 아주 단순 FC: pkj_j = sum_r s_grc[r] * alpha
    # 진짜 weight가 있으면 이 부분을 w에서 가져와 곱해라
    alpha = 0.1
    for j in range(N_PKJ):
        v_pkj[j] += alpha * np.sum(s_grc)   # input current
        v_pkj[j] -= v_pkj[j] / TAU_PKJ
        if v_pkj[j] >= THRESHOLD:
            s_pkj[j] = 1
            v_pkj[j] = 0.0
        else:
            s_pkj[j] = 0

    # ------------------------------------------------------------
    # 6) pkj -> motor (4개)
    # ------------------------------------------------------------
    for m in range(N_MOTOR):
        # pkj 스파이크 총합을 모터 입력으로
        I_m = np.sum(s_pkj) * 0.2
        v_motor[m] += I_m
        v_motor[m] -= v_motor[m] / TAU_MOTOR
        # 모터 출력은 그냥 전위로 둠
        v_motor_output[m] = v_motor[m]

    # ------------------------------------------------------------
    # 7) 이 스텝에서의 action 결정
    #    지금은 그냥 motor 전위 부호로 액션 만들어서 env에 다시 넣게
    # ------------------------------------------------------------
    action = v_motor_output

    # 종료 체크
    if terminated or truncated:
        break
    env.close_viewer()
