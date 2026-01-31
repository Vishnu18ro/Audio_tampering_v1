import os
import random
import librosa
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# input clean file
input_path = os.path.join(
    BASE_DIR, "..", "data", "authentic_wav", "auth_03.wav"
)

# output folder
output_dir = os.path.join(
    BASE_DIR, "..", "data", "manipulated", "tampered"
)
os.makedirs(output_dir, exist_ok=True)

# load audio
audio, sr = librosa.load(input_path, sr=None)

# choose random speed factor
# slow: 0.7–0.9 | fast: 1.1–1.3
if random.random() < 0.5:
    speed = random.uniform(0.7, 0.9)
else:
    speed = random.uniform(1.1, 1.3)

# apply time stretching
tampered_audio = librosa.effects.time_stretch(audio, rate=speed)

# save output
output_path = os.path.join(output_dir, "tamper_speed_auto_01.wav")
sf.write(output_path, tampered_audio, sr)

print("Random speed change created:")
print("Speed factor:", round(speed, 3))
print("Saved to:", output_path)
