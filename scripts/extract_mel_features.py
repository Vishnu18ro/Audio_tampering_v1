import os
import csv
import librosa
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "..", "data", "dataset.csv")
FEATURE_DIR = os.path.join(BASE_DIR, "..", "data", "features")

os.makedirs(FEATURE_DIR, exist_ok=True)

# Mel parameters (for CNN-friendly input)
SR = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_FRAMES = 300  # fixed time dimension

def extract_mel(path):
    y, sr = librosa.load(path, sr=SR)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def pad_or_trim(mel):
    if mel.shape[1] < TARGET_FRAMES:
        pad = TARGET_FRAMES - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant")
    else:
        mel = mel[:, :TARGET_FRAMES]
    return mel

with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        audio_path = os.path.join(BASE_DIR, "..", row["filepath"])
        mel = extract_mel(audio_path)
        mel = pad_or_trim(mel)

        name = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(FEATURE_DIR, name + ".npy")

        np.save(out_path, mel)

print("✔ Mel feature extraction completed.")
