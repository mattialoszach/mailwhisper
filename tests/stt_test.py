from utils.mic_mem import record_until_enter_mem
from utils.stt_whisper_mem import transcribe_array

audio = record_until_enter_mem(samplerate=16000)
text, detected_lang = transcribe_array(audio, samplerate=16000)

print(f"📝 Transkript ({detected_lang or 'auto'}): {text}")