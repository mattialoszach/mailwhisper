import sys, threading, queue
from typing import Optional, List
import numpy as np
import sounddevice as sd

class ButtonControlledRecorder:
    def __init__(self, samplerate: int = 16000, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._chunks: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._running = False

    def _audio_cb(self, indata, frames, time, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        self._q.put(indata.copy())

    def start(self):
        if self._running:
            return
        self._running = True
        self._chunks.clear()
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self._audio_cb,
            dtype="float32",
        )
        self._stream.start()
        # Background thread collects frames
        def pump():
            while self._running:
                try:
                    self._chunks.append(self._q.get(timeout=0.1))
                except queue.Empty:
                    pass
        self._pump_thread = threading.Thread(target=pump, daemon=True)
        self._pump_thread.start()

    def stop(self) -> np.ndarray:
        if not self._running:
            return np.zeros((0, self.channels), dtype=np.float32)
        self._running = False
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
        if not self._chunks:
            raise RuntimeError("No audio recorded.")
        return np.concatenate(self._chunks, axis=0)
