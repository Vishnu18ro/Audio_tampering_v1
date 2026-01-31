import os
import random
import librosa
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# input clean file
input_path = os.path.join(
    BASE_DIR, "..", "data", "authentic_wav", "auth_02.wav"
)

# output folder
output_dir = os.path.join(
    BASE_DIR, "..", "data", "manipulated", "tampered"
)
os.makedirs(output_dir, exist_ok=True)

# load audio
audio, sr = librosa.load(input_path, sr=None)
total_len = len(audio)

# choose random splice length (3%–8%)
splice_len = random.randint(int(0.03 * total_len), int(0.08 * total_len))

# choose source segment
src_start = random.randint(int(0.1 * total_len), total_len - splice_len - 1)
segment = audio[src_start:src_start + splice_len]

# choose insertion point (different region)
insert_pos = random.randint(int(0.1 * total_len), int(0.9 * total_len))

# perform splicing
tampered_audio = (
    list(audio[:insert_pos]) +
    list(segment) +
    list(audio[insert_pos:])
)

# save output
output_path = os.path.join(output_dir, "tamper_splice_auto_01.wav")
sf.write(output_path, tampered_audio, sr)

print("Random splicing created:")
print("Splice length:", splice_len)
print("Inserted at position:", insert_pos)
print("Saved to:", output_path)
