from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable

import numpy as np

from config import BotConfig


@dataclass(frozen=True)
class BiteResult:
    detected: bool
    peak_rms: float
    baseline_rms: float
    threshold: float


def calculate_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    values = samples.astype(np.float64)
    return float(np.sqrt(np.mean(values * values)))


def calculate_threshold(baseline: float, multiplier: float, minimum: float) -> float:
    return max(minimum, baseline * multiplier)


class AudioDetector:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._audio: Any = None
        self._stream: Any = None

    def _open(self) -> None:
        import pyaudiowpatch as pyaudio
        self._audio = pyaudio.PyAudio()
        devices = list(self._audio.get_loopback_device_info_generator())
        if self.config.audio_device_name:
            devices = [d for d in devices if self.config.audio_device_name.lower() in d["name"].lower()]
        if not devices:
            raise RuntimeError("No matching WASAPI loopback device found")
        device = devices[0]
        self._stream = self._audio.open(
            format=pyaudio.paInt16, channels=max(1, int(device["maxInputChannels"])),
            rate=int(device["defaultSampleRate"]), input=True,
            input_device_index=int(device["index"]), frames_per_buffer=self.config.audio_chunk_size,
        )

    def wait_for_bite(self, timeout: float, stop_event: threading.Event,
                      sample_callback: Callable[[float, float, float, bool], None] | None = None) -> BiteResult:
        self._open()
        calibration: list[float] = []
        peak = 0.0
        spikes = 0
        started = time.monotonic()
        baseline = threshold = 0.0
        try:
            while time.monotonic() - started < timeout and not stop_event.is_set():
                data = self._stream.read(self.config.audio_chunk_size, exception_on_overflow=False)
                rms = calculate_rms(np.frombuffer(data, dtype=np.int16))
                peak = max(peak, rms)
                if len(calibration) < self.config.audio_calibration_chunks:
                    calibration.append(rms)
                    baseline = float(np.median(calibration))
                    threshold = calculate_threshold(baseline, self.config.audio_threshold_multiplier,
                                                    self.config.minimum_audio_threshold)
                    continue
                is_spike = rms >= threshold
                spikes = spikes + 1 if is_spike else 0
                if sample_callback:
                    sample_callback(rms, baseline, threshold, is_spike)
                if spikes >= self.config.audio_consecutive_spikes:
                    return BiteResult(True, peak, baseline, threshold)
            return BiteResult(False, peak, baseline, threshold)
        finally:
            self.close()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            finally:
                self._stream = None
        if self._audio is not None:
            self._audio.terminate()
            self._audio = None
