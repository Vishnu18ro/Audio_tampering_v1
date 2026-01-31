import os
import csv
import random
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "dataset.csv")
FEATURE_DIR = os.path.join(BASE_DIR, "..", "data", "features")

WINDOW = 40
HOP = 20
FILE_THRESH = 0.50   # file-level voting (fixed, correct)

# -------------------------------------------------
# STEP 1: LOAD FILE-LEVEL DATA
# -------------------------------------------------
files = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = os.path.splitext(os.path.basename(row["filepath"]))[0]
        mel = np.load(os.path.join(FEATURE_DIR, name + ".npy"))
        label = int(row["label"])
        base = name.replace("del_", "").replace("splice_", "").replace("speed_", "")
        files.append((base, mel, label))

# -------------------------------------------------
# STEP 2: BALANCED FILE-LEVEL SPLIT
# -------------------------------------------------
clean_files = [f for f in files if f[2] == 0]
tamper_files = [f for f in files if f[2] == 1]

random.seed(42)
random.shuffle(clean_files)
random.shuffle(tamper_files)

N = min(len(clean_files), len(tamper_files), 5)

test_files = clean_files[:N] + tamper_files[:N]
train_files = clean_files[N:] + tamper_files[N:]

print("Train files:", len(train_files))
print("Test files:", len(test_files))

# -------------------------------------------------
# STEP 3: WINDOWING FUNCTION
# -------------------------------------------------
def make_windows(file_list):
    X, y, bases = [], [], []
    for base, mel, label in file_list:
        T = mel.shape[1]
        if T < WINDOW:
            win = np.pad(mel, ((0,0),(0, WINDOW-T)))
            X.append(win)
            y.append(label)
            bases.append(base)
        else:
            for i in range(0, T - WINDOW + 1, HOP):
                X.append(mel[:, i:i+WINDOW])
                y.append(label)
                bases.append(base)
    return np.array(X)[..., np.newaxis], np.array(y), np.array(bases)

X_all, y_all, _ = make_windows(train_files)

# -------------------------------------------------
# STEP 4: TRAIN / VALIDATION SPLIT (WINDOW LEVEL)
# -------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

print("Train windows:", len(X_train))
print("Val windows:", len(X_val))

# -------------------------------------------------
# STEP 5: MODEL
# -------------------------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128, WINDOW, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    GlobalAveragePooling2D(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# -------------------------------------------------
# STEP 6: AUTO-LEARN WINDOW THRESHOLD (YOUDEN J)
# -------------------------------------------------
val_probs = model.predict(X_val).flatten()
fpr, tpr, thresholds = roc_curve(y_val, val_probs)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
WIN_THRESH = thresholds[best_idx]

print(f"\nLearned WIN_THRESH = {WIN_THRESH:.3f}")

# -------------------------------------------------
# STEP 7: FILE-LEVEL EVALUATION
# -------------------------------------------------
X_test, y_test, base_test = make_windows(test_files)

file_votes = defaultdict(list)
file_gt = {}

for win, gt, base in zip(X_test, y_test, base_test):
    prob = model.predict(win[None, ...], verbose=0)[0][0]
    file_votes[base].append(int(prob > WIN_THRESH))
    file_gt[base] = gt

y_file_pred, y_file_true = [], []
for b in file_votes:
    ratio = sum(file_votes[b]) / len(file_votes[b])
    y_file_pred.append(int(ratio >= FILE_THRESH))
    y_file_true.append(file_gt[b])

acc = accuracy_score(y_file_true, y_file_pred)
cm = confusion_matrix(y_file_true, y_file_pred)

print("\nFILE-LEVEL Test Accuracy:", acc)
print("Confusion Matrix:\n", cm)
model.save("model.h5")
