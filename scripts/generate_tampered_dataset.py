import os
import random
import librosa
import soundfile as sf

# Base directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# IMPORTANT: use FIXED clean audio
CLEAN_DIR = os.path.join(BASE_DIR, "..", "data", "authentic_fixed")

# Output directory for tampered audio
OUT_DIR = os.path.join(BASE_DIR, "..", "data", "manipulated", "tampered")
os.makedirs(OUT_DIR, exist_ok=True)


# -------- Tampering Functions (SUBTLE) --------

def random_deletion(audio):
    total = len(audio)
    # subtle deletion: 2%–6%
    del_len = random.randint(int(0.02 * total), int(0.06 * total))
    start = random.randint(int(0.1 * total), total - del_len - 1)
    return audio[:start].tolist() + audio[start + del_len:].tolist()


def random_splicing(audio):
    total = len(audio)
    # subtle splice: 2%–5%
    seg_len = random.randint(int(0.02 * total), int(0.05 * total))
    src_start = random.randint(int(0.1 * total), total - seg_len - 1)
    segment = audio[src_start:src_start + seg_len]

    insert_pos = random.randint(int(0.1 * total), int(0.9 * total))
    return audio[:insert_pos].tolist() + segment.tolist() + audio[insert_pos:].tolist()


def random_speed(audio):
    # subtle speed change
    if random.random() < 0.5:
        rate = random.uniform(0.9, 0.97)
    else:
        rate = random.uniform(1.03, 1.1)

    stretched = librosa.effects.time_stretch(y=audio, rate=rate)
    return stretched, rate


# -------- Main Loop --------

clean_files = sorted([f for f in os.listdir(CLEAN_DIR) if f.endswith(".wav")])

print("Total clean files:", len(clean_files))

for idx, file in enumerate(clean_files, start=1):
    audio_path = os.path.join(CLEAN_DIR, file)
    audio, sr = librosa.load(audio_path, sr=None)

    base_name = os.path.splitext(file)[0]

    # 1 × deletion
    out_audio = random_deletion(audio)
    out_name = f"del_{base_name}.wav"
    sf.write(os.path.join(OUT_DIR, out_name), out_audio, sr)

    # 1 × splicing
    out_audio = random_splicing(audio)
    out_name = f"splice_{base_name}.wav"
    sf.write(os.path.join(OUT_DIR, out_name), out_audio, sr)

    # 1 × speed
    out_audio, rate = random_speed(audio)
    out_name = f"speed_{base_name}.wav"
    sf.write(os.path.join(OUT_DIR, out_name), out_audio, sr)

    print(f"[{idx}/{len(clean_files)}] processed {file}")

print("✔ Subtle tampered dataset generation completed.")
