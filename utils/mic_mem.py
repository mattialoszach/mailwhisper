import sys, threading, queue
import numpy as np
import sounddevice as sd

def record_until_enter_mem(samplerate: int = 16000, channels: int = 1) -> np.ndarray:
    """
    Records audio until the user presses Enter.
    """
    q = queue.Queue()
    stop = threading.Event()

    def on_Enter():
        input("") # ENTER terminates
        stop.set()

    def audio_cb(indata, frames, time, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        q.put(indata.copy())

    threading.Thread(target=on_Enter, daemon=True).start()

    print("🎙️ Recording… (press ENTER to stop)")
    chunks = []
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_cb, dtype='float32'):
        while not stop.is_set():
            try:
                chunks.append(q.get(timeout=0.1))
            except queue.Empty:
                pass
    
    if not chunks:
        raise RuntimeError("No Audiodata recorded.")
    audio = np.concatenate(chunks, axis=0)
    return audio