import os
import soundfile as sf

# get path of THIS script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_dir = os.path.join(BASE_DIR, "..", "data", "authentic")
output_dir = os.path.join(BASE_DIR, "..", "data", "authentic_wav")

os.makedirs(output_dir, exist_ok=True)

count = 1
for file in os.listdir(input_dir):
    if file.lower().endswith(".flac"):
        audio, sr = sf.read(os.path.join(input_dir, file))
        out_name = f"auth_{count:02d}.wav"
        sf.write(os.path.join(output_dir, out_name), audio, sr)
        count += 1

print("FLAC to WAV conversion done.")
