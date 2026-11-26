import json
import random
from pathlib import Path

INPUT_FILE = "datasets/code_evol_scot_pilot_v2.json"

TRAIN_FILE = "datasets/evol_scot_train.json"
VALID_FILE = "datasets/evol_scot_valid.json"

TRAIN_RATIO = 0.9
SEED = 42


def make_text(sample):
    return (
        "### Programming Problem\n"
        + sample["enhanced_instruction"].strip()
        + "\n\n"
        "### Reasoning\n"
        + sample["reasoning"].strip()
        + "\n\n"
        "### Pseudo-code\n"
        + sample["pseudocode"].strip()
        + "\n\n"
        "### Final Code\n"
        + sample["final_code"].strip()
    )


def main():

    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    print("Loaded samples:", len(data))

    records = []

    for sample in data:

        text = make_text(sample)

        records.append({
            "instruction": sample["enhanced_instruction"].strip(),
            "output": (
                "Reasoning:\n"
                + sample["reasoning"].strip()
                + "\n\n"
                "Pseudo-code:\n"
                + sample["pseudocode"].strip()
                + "\n\n"
                "Final Code:\n"
                + sample["final_code"].strip()
            ),
            "text": text
        })

    random.seed(SEED)
    random.shuffle(records)

    split = int(len(records) * TRAIN_RATIO)

    train = records[:split]
    valid = records[split:]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)

    with open(VALID_FILE, "w", encoding="utf-8") as f:
        json.dump(valid, f, indent=2, ensure_ascii=False)

    print("\nDataset preparation completed!")
    print("Total     :", len(records))
    print("Training  :", len(train))
    print("Validation:", len(valid))
    print("Train     :", TRAIN_FILE)
    print("Valid     :", VALID_FILE)


if __name__ == "__main__":
    main()
