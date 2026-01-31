import os
import librosa

# get path of THIS script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

audio_path = os.path.join(
    BASE_DIR, "..", "data", "authentic_wav", "auth_01.wav"
)

audio, sr = librosa.load(audio_path, sr=None)

print("Sample rate:", sr)
print("Audio shape:", audio.shape)
print("Duration (seconds):", len(audio) / sr)
