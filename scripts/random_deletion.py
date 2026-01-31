import os
import random
import librosa
import soundfile as sf

# get base directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# input clean file
input_path = os.path.join(
    BASE_DIR, "..", "data", "authentic_wav", "auth_01.wav"
)

# output folder
output_dir = os.path.join(
    BASE_DIR, "..", "data", "manipulated", "tampered"
)
os.makedirs(output_dir, exist_ok=True)

# load audio
audio, sr = librosa.load(input_path, sr=None)

# choose random deletion length (5%–15% of audio)
total_len = len(audio)
del_len = random.randint(int(0.05 * total_len), int(0.15 * total_len))

# choose random start point (avoid very start/end)
start = random.randint(int(0.1 * total_len), total_len - del_len - 1)

# perform deletion
tampered_audio = list(audio[:start]) + list(audio[start + del_len:])

# save output
output_path = os.path.join(output_dir, "tamper_del_auto_01.wav")
sf.write(output_path, tampered_audio, sr)

print("Random deletion created:")
print("Deleted samples:", del_len)
print("Saved to:", output_path)
