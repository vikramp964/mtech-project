import json
import random
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
SEED = 42
TRAIN_RATIO = 0.95

DATA_DIR = Path("datasets")
INPUT_FILE = DATA_DIR / "code_alpaca_20k.json"
TRAIN_FILE = DATA_DIR / "train.json"
VALID_FILE = DATA_DIR / "valid.json"

# -----------------------------
# Load dataset
# -----------------------------
print(f"Loading dataset from: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# -----------------------------
# Shuffle dataset
# -----------------------------
random.seed(SEED)
random.shuffle(data)

# -----------------------------
# Train / Validation Split
# -----------------------------
split_index = int(len(data) * TRAIN_RATIO)

train_data = data[:split_index]
valid_data = data[split_index:]

# -----------------------------
# Save datasets
# -----------------------------
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open(VALID_FILE, "w", encoding="utf-8") as f:
    json.dump(valid_data, f, indent=2, ensure_ascii=False)

# -----------------------------
# Summary
# -----------------------------
print("\nDataset preparation completed successfully!")
print(f"Training samples   : {len(train_data)}")
print(f"Validation samples : {len(valid_data)}")
print(f"Train file         : {TRAIN_FILE}")
print(f"Validation file    : {VALID_FILE}")
