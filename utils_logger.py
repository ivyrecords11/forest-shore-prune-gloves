from __future__ import annotations

from config import SimulationConfig
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import matplotlib
# Force a non-interactive backend so plotting works on servers without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

cfg = SimulationConfig()
DEBUG = False

"""Utility classes for logging spike and potential activity during training.
The original project relied on small helper classes that plotted data using
matplotlib's default interactive backend.  When the training script was run in
headless environments (such as remote servers or automated tests) the plotting
calls attempted to open an interactive window, causing the Python process to
terminate.
This module provides light‑weight replacements that record the incoming data and
export figures directly to files using matplotlib's non‑interactive ``Agg``
backend.  The API matches the usage in the training script: ``save_spikes`` is
called repeatedly to append data for the current episode and ``plot`` is called
afterwards to write the figure to ``plot_name``.  ``clear`` simply discards the
stored samples so that the same logger instance can be reused across episodes.
If file creation fails (for example because the path is invalid) the error is
caught and logged, but the training loop continues – preventing plotting issues
from aborting the run.
"""
'''
@dataclass
class _BaseLogger:
    plot_name: str
    plot_size: Optional[tuple[float, float]] = None
    def __post_init__(self) -> None:
        self.path = Path(self.plot_name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[np.ndarray] = []
    # The concrete subclasses expose ``save_spikes``/``save_potentials`` – both
    # names map to this helper to keep the interface flexible.
    def _append(self, values: Iterable[float] | np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float32)
        self._records.append(arr.copy())
        if DEBUG:
            print(f"[LOGGER] appended sample with shape {arr.shape}")
    def clear(self) -> None:
        if DEBUG:
            print("[LOGGER] clearing stored records")
        self._records.clear()
    # ``plot`` is implemented by the subclasses.
    def plot(self) -> None:  # pragma: no cover - virtual method
        raise NotImplementedError
    # Shared utility to save a matplotlib figure without interrupting training.
    def _save_figure(self, fig: plt.Figure) -> None:
        try:
            fig.savefig(self.path, bbox_inches="tight")
            if DEBUG:
                print(f"[LOGGER] saved plot to {self.path}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[LOGGER] failed to save plot '{self.path}': {exc}")
        finally:
            plt.close(fig)
class SpikePlotter(_BaseLogger):
    """Visualises spike trains as a heatmap (time on the x-axis)."""
    def save_spikes(self, spikes: Iterable[float] | np.ndarray) -> None:
        self._append(spikes)
    def plot(self) -> None:
        if not self._records:
            if DEBUG:
                print("[LOGGER] no spike data recorded – skipping plot")
            return
        data = np.stack(self._records, axis=0)
        if data.ndim == 1:
            data = data[:, None]
        fig, ax = plt.subplots(figsize=self.plot_size or (12, 4))
        im = ax.imshow(
            data.T,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap="viridis",
        )
        ax.set_xlabel("Time step")
        ax.set_ylabel("Unit index")
        ax.set_title("Spike activity")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Spike value")
        self._save_figure(fig)
class SpikeHeatmap(_BaseLogger):
    """Aggregates sensor activity into a 2D heatmap."""
    def __init__(self, *args, shape: tuple[int, int], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shape = shape
    def save_spikes(self, spikes: Iterable[float] | np.ndarray) -> None:
        self._append(spikes)
    def plot(self) -> None:
        if not self._records:
            if DEBUG:
                print("[LOGGER] no heatmap data recorded – skipping plot")
            return
        data = np.stack(self._records, axis=0)
        try:
            data = data.reshape((-1, *self.shape))
        except ValueError:
            # If reshaping fails we fall back to averaging the flattened data.
            data = data.reshape(len(self._records), -1)
            if DEBUG:
                print("[LOGGER] heatmap reshape failed, falling back to flat view")
            avg = data.mean(axis=0)
            fig, ax = plt.subplots(figsize=self.plot_size or (6, 6))
            im = ax.imshow(avg.reshape(1, -1), aspect="auto", cmap="hot")
        else:
            avg = data.mean(axis=0)
            fig, ax = plt.subplots(figsize=self.plot_size or (6, 6))
            im = ax.imshow(avg, origin="lower", cmap="hot")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        ax.set_title("Average sensor activity")
        fig.colorbar(im, ax=ax, shrink=0.8)
        self._save_figure(fig)
class PotentialPlotter(_BaseLogger):
    """Plots membrane potentials for one or multiple neurons."""
    def save_potentials(self, potentials: Iterable[float] | np.ndarray) -> None:
        self._append(potentials)
    # Alias used in the training script.
    save_spikes = save_potentials
    def plot(self) -> None:
        if not self._records:
            if DEBUG:
                print("[LOGGER] no potential data recorded – skipping plot")
            return
        data = np.stack(self._records, axis=0)
        if data.ndim == 1:
            data = data[:, None]
        fig, ax = plt.subplots(figsize=self.plot_size or (12, 4))
        ax.plot(data)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Membrane potential")
        ax.set_title("Neuron potentials")
        self._save_figure(fig)
class PotentialPlotter_vertical(PotentialPlotter):
    """Same as :class:`PotentialPlotter` but arranged vertically."""
    def plot(self) -> None:
        if not self._records:
            if DEBUG:
                print("[LOGGER] no potential data recorded – skipping plot")
            return
        data = np.stack(self._records, axis=0)
        if data.ndim == 1:
            data = data[:, None]
        n_series = data.shape[1]
        fig, axes = plt.subplots(
            n_series,
            1,
            figsize=self.plot_size or (8, 2 * n_series),
            sharex=True,
        )
        if n_series == 1:
            axes = (axes,)
        for idx, ax in enumerate(axes):
            ax.plot(data[:, idx])
            ax.set_ylabel(f"Unit {idx}")
        axes[-1].set_xlabel("Time step")
        fig.suptitle("Neuron potentials")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        self._save_figure(fig)
__all__ = [
    "SpikePlotter",
    "SpikeHeatmap",
    "PotentialPlotter",
    "PotentialPlotter_vertical",
]

'''
class SpikePlotter():
    def __init__(self, plot_name: str, DEBUG = False, plot_size = (15, 5)):
        self.spike_log = []
        self.DEBUG = DEBUG
        self.plot_name = plot_name
    
    def clear(self):
        self.spike_log = []

    def save_spikes(self, spikes: np.ndarray):
        if self.DEBUG: 
            print(spikes)
        if type(spikes) == list:
            spikes = np.array(spikes, dtype=np.int32)
        spikes = spikes.flatten()
        self.spike_log.append(spikes)
    def plot(self, visualize = False):
        #print(self.spike_log)
        spike_mat = np.stack(self.spike_log, axis=0)          # [T, 100], 0/1
        spike_mat = spike_mat.T                           # [100, T]로 전치: 채널(y), 시간(x)

        plt.figure(figsize=(15, 5))
        # 이진 행렬을 래스터처럼 시각화. time step이 x축, 채널이 y축
        plt.imshow(spike_mat, aspect='auto', interpolation='nearest', origin='lower')
        plt.xlabel('Time step')
        plt.ylabel('Input channel')
        plt.title(f'{self.plot_name} Spike Plot')
        #plt.colorbar(label='Spike (0/1)')
        plt.tight_layout()
        plt.savefig(f"{self.plot_name}.png")
        plt.close('all')
        if visualize: plt.show()

class PotentialPlotter():
    def __init__(self, plot_name: str, DEBUG = False, plot_size = (15, 5)):
        self.spike_log = []
        self.DEBUG = DEBUG
        self.plot_name = plot_name
        self.plot_size = plot_size
    
    def clear(self):
        self.spike_log = []

    def save_spikes(self, spikes: np.ndarray):
        if self.DEBUG: 
            #print(spikes)
            self.DEBUG = False
        #if type(spikes) ==np.ndarray:
        if type(spikes) == list:
            spikes = np.array(spikes, dtype=np.float32)
        spikes = spikes.flatten()
        self.spike_log.append(spikes)

    def plot(self, visualize = False):
        spike_mat = np.stack(self.spike_log, axis=0)          # [T, 100], 0/1
        spike_mat = spike_mat.T                           # [100, T]로 전치: 채널(y), 시간(x)

        plt.figure(figsize=self.plot_size)
        # 이진 행렬을 래스터처럼 시각화. time step이 x축, 채널이 y축
        plt.imshow(spike_mat, aspect='auto', interpolation='nearest', origin='lower')
        plt.xlabel('Time step')
        plt.ylabel('Input channel')
        plt.title(f'{self.plot_name} Potential Plot')
        #plt.colorbar(label='Spike (0/1)')
        plt.tight_layout()
        plt.savefig(f"{self.plot_name}.png")
        plt.close('all')
        if visualize: plt.show()

class PotentialPlotter_vertical():
    def __init__(self, plot_name: str, DEBUG = False, plot_size = (15, 20), n_channels = cfg.n_grc):
        self.spike_log = []
        self.DEBUG = DEBUG
        self.plot_name = plot_name
        self.plot_size = plot_size
        self.n_channels = n_channels
    
    def clear(self):
        self.spike_log = []

    def save_spikes(self, spikes: np.ndarray):
        if self.DEBUG: 
            #print(spikes)
            self.DEBUG = False
        #if type(spikes) ==np.ndarray:
        if type(spikes) == list:
            spikes = np.array(spikes, dtype=np.float32)
        spikes = spikes.flatten()
        self.spike_log.append(spikes)

    def plot(self, visualize = False):
        spike_mat = np.stack(self.spike_log, axis=0)          # [T, 6*], 0/1
        spike_mat = spike_mat.T                           # [100, T]로 전치: 채널(y), 시간(x)
        
        plt.figure(figsize=self.plot_size)
        plt.xlabel('Time step')
        plt.ylabel('Input channel')
        plt.title(f'{self.plot_name} Potential Plot')
        for i in range(self.n_channels):
            plt.subplot(self.n_channels, 1, i)
            plt.imshow(spike_mat[i*36:(i+1)*36], aspect='auto', interpolation='nearest', origin='lower')
            
        # 이진 행렬을 래스터처럼 시각화. time step이 x축, 채널이 y축
        #plt.colorbar(label='Spike (0/1)')
        plt.tight_layout()
        plt.savefig(f"{self.plot_name}.png")
        if visualize: plt.show()
        plt.close('all')

from typing import Sequence, Tuple, Optional, Union

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]

class PotentialPlotterVertical:
    """
    - 매 타임스텝 스파이크를 (H,W) 또는 (N,)로 받아 누적
    - plot():
        1) 전체 채널 라스터(채널×시간) 1장
        2) 선택한 채널들의 세로 스택 라인 플롯 (옵션)
    """
    def __init__(
        self,
        plot_name: str,
        grid_shape: Tuple[int, int] = (6, 6),
        channels_to_plot: Optional[Sequence[int]] = None,  # 라인 플롯으로 볼 채널 인덱스
        DEBUG: bool = False,
    ):
        self.plot_name = plot_name
        self.H, self.W = grid_shape
        self.N = self.H * self.W
        self.DEBUG = DEBUG
        self.spike_log: list[np.ndarray] = []  # [T, N]으로 누적
        if channels_to_plot is None:
            # 기본: 앞 8개만 라인 플롯 (너무 많으면 그림이 기형적으로 커짐)
            self.channels_to_plot = list(range(min(8, self.N)))
        else:
            self.channels_to_plot = list(channels_to_plot)

    def clear(self):
        self.spike_log.clear()

    def save_spikes(self, spikes):
        """
        허용 형태:
        - (H, W)
        - (N,)
        - (T, H, W)
        - (T, N)
        값: 0/1, bool, 또는 연속값(포텐셜) OK
        """
        # torch.Tensor -> numpy
        if "torch" in str(type(spikes)):
            import torch
            if isinstance(spikes, torch.Tensor):
                spikes = spikes.detach().cpu().numpy()

        x = np.asarray(spikes)
        # bool -> float32
        if x.dtype == np.bool_:
            x = x.astype(np.float32)
        else:
            x = x.astype(np.float32, copy=False)

        # 내부 헬퍼: (N,)만 append
        def _append_1d(v):
            if v.ndim == 2:  # (H,W) -> (N,)
                if v.shape != (self.H, self.W):
                    raise ValueError(f"Expected {(self.H, self.W)} but got {v.shape}")
                v = v.reshape(-1)
            elif v.ndim == 1:
                if v.size != self.N:
                    raise ValueError(f"Expected length {self.N} but got {v.size}")
            else:
                raise ValueError(f"Single sample must be 1D or 2D, got {v.ndim}D")
            self.spike_log.append(v)

        if x.ndim == 3:
            # (T, H, W)
            if x.shape[1:] == (self.H, self.W):
                for t in range(x.shape[0]):
                    _append_1d(x[t])
            else:
                raise ValueError(f"Expected (T,{self.H},{self.W}) but got {x.shape}")
        elif x.ndim == 2:
            # (T, N) 또는 (H, W)
            if x.shape == (self.H, self.W):
                _append_1d(x)
            elif x.shape[1] == self.N:
                # (T, N)
                for t in range(x.shape[0]):
                    _append_1d(x[t])
            else:
                raise ValueError(
                    f"2D input must be (H,W) or (T,N). Got {x.shape}, N={self.N}"
                )
        elif x.ndim == 1:
            _append_1d(x)
        else:
            raise ValueError(f"Unsupported ndim {x.ndim}")

        if self.DEBUG:
            print(f"[save_spikes] appended {len(self.spike_log)} frames. last shape=(N,) with N={self.N}")
            self.DEBUG = False


        def plot(self, visualize: bool = False, dpi: int = 120):
            if not self.spike_log:
                raise RuntimeError("No spikes saved. Call save_spikes() first.")

            # [T, N] → (N, T)
            S = np.stack(self.spike_log, axis=0)       # (T, N)
            S = S.T                                    # (N, T)

            # --- (1) 라스터 플롯 ---
            plt.figure(figsize=(10, 4), dpi=dpi)
            # imshow에선 0~1 값을 그대로 intensity로 쓰면 ‘흰/검’ 라스터가 됨
            plt.imshow(S, aspect='auto', interpolation='nearest', origin='lower')
            plt.colorbar(label='spike/potential')
            plt.ylabel('channel (0..N-1)')
            plt.xlabel('time step')
            plt.title(f'{self.plot_name} — Raster (N={self.N}, T={S.shape[1]})')
            plt.tight_layout()
            plt.savefig(f"{self.plot_name}_raster.png")
            if visualize:
                plt.show()
            plt.close()

            # --- (2) 선택 채널 라인 플롯 (세로 스택) ---
            K = len(self.channels_to_plot)
            if K > 0:
                fig, axes = plt.subplots(
                    nrows=K, ncols=1, figsize=(10, 2.0 * K), dpi=dpi, sharex=True
                )
                if K == 1:
                    axes = [axes]
                T = S.shape[1]
                t = np.arange(T)
                for ax, ch in zip(axes, self.channels_to_plot):
                    if ch < 0 or ch >= self.N:
                        raise IndexError(f"channel index {ch} out of range [0,{self.N})")
                    ax.plot(t, S[ch])
                    ax.set_ylabel(f'ch {ch}')
                axes[0].set_title(f'{self.plot_name} — Selected channels')
                axes[-1].set_xlabel('time step')
                plt.tight_layout()
                plt.savefig(f"{self.plot_name}_lines.png")
                if visualize:
                    plt.show()
                plt.close()

class SpikeHeatmap:
    def __init__(self, plot_name: str, shape: tuple = (cfg.n_sensor_1d, cfg.n_sensor_1d), DEBUG = False):
        self.spike_acc = np.zeros(shape, dtype=np.int32)
        self.shape = shape
        self.plot_name = plot_name
    
    def clear(self):
        self.spike_log = []

    def save_spikes(self, spikes):
        spikes = np.array(spikes, dtype = np.int32)
        spikes.reshape(self.shape)
        self.spike_acc += spikes
    
    def plot(self, visualize = False):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.spike_acc, cmap='hot', interpolation='nearest', vmin=0, vmax=int(cfg.max_firing_rate*1.2))
        plt.colorbar(label='Spike Intensity')
        plt.title(f"{self.plot_name} Heatmap")
        plt.tight_layout()
        plt.savefig(f"{self.plot_name}.png")
        if visualize: plt.show()
        plt.close('all')
