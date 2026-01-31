import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FIXED clean audio
CLEAN_DIR = os.path.join(BASE_DIR, "..", "data", "authentic_fixed")

# Subtle tampered audio
TAMPER_DIR = os.path.join(BASE_DIR, "..", "data", "manipulated", "tampered")

CSV_PATH = os.path.join(BASE_DIR, "..", "data", "dataset.csv")

rows = []

# Clean files → label 0
for file in sorted(os.listdir(CLEAN_DIR)):
    if file.endswith(".wav"):
        rows.append([f"data/authentic_fixed/{file}", 0])

# Tampered files → label 1
for file in sorted(os.listdir(TAMPER_DIR)):
    if file.endswith(".wav"):
        rows.append([f"data/manipulated/tampered/{file}", 1])

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label"])
    writer.writerows(rows)

print("✔ dataset.csv created")
print("Clean samples:", len([r for r in rows if r[1] == 0]))
print("Tampered samples:", len([r for r in rows if r[1] == 1]))
print("Total samples:", len(rows))
