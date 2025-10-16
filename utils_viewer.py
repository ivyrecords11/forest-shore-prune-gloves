# viewer_utils.py
import time
import mujoco
import mujoco.viewer as viewer

class LiveViewer:
    def __init__(self, model, data):
        # launch_passive: 코드 진행을 막지 않음
        self.v = viewer.launch_passive(model, data)  # context manager도 가능
        # 필요하면 self.v.cam / self.v.opt 등을 수정 (lock 필수)

    def sync(self):
        # 시뮬 한 스텝 후 호출하면 프레임 갱신
        self.v.sync()

    def close(self):
        self.v.close()
