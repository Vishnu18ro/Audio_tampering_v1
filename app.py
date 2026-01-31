import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# ---------------- CONFIG ----------------
MODEL_PATH = "model.h5"
WIN_THRESH = 0.695
FILE_THRESH = 0.50
WINDOW = 40
HOP = 20
SR = 16000

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

def extract_mel(path):
    y, _ = librosa.load(path, sr=SR)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=128, n_fft=2048, hop_length=512
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel

def predict_file(audio_path):
    mel = extract_mel(audio_path)
    T = mel.shape[1]
    votes = []

    if T < WINDOW:
        mel = np.pad(mel, ((0,0),(0, WINDOW-T)))
        mel = mel[..., None][None, ...]
        p = model.predict(mel, verbose=0)[0][0]
        votes.append(p > WIN_THRESH)
    else:
        for i in range(0, T - WINDOW + 1, HOP):
            win = mel[:, i:i+WINDOW]
            win = win[..., None][None, ...]
            p = model.predict(win, verbose=0)[0][0]
            votes.append(p > WIN_THRESH)

    ratio = sum(votes) / len(votes)
    return ratio

# ---------------- UI ----------------
st.title("🎧 Audio Tampering Detection")

uploaded = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded.read())

    st.audio("temp.wav")

    if st.button("Analyze"):
        score = predict_file("temp.wav")

        if score >= FILE_THRESH:
            st.error(f"⚠️ TAMPERED detected (score: {score:.2f})")
        else:
            st.success(f"✅ CLEAN audio (score: {score:.2f})")
