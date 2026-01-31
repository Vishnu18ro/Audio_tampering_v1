import os
import librosa
import soundfile as sf
import numpy as np

INPUT_DIR = r"data/authentic_wav"
OUTPUT_DIR = r"data/authentic_fixed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 16000
DURATION = 5
SAMPLES = SR * DURATION

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".wav"):
        continue

    path = os.path.join(INPUT_DIR, fname)
    y, sr = librosa.load(path, sr=SR, mono=True)

    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = np.pad(y, (0, SAMPLES - len(y)))

    sf.write(os.path.join(OUTPUT_DIR, fname), y, SR)

print("✅ clean audio fixed to 5 seconds")
