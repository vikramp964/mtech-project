import json

ORIGINAL_FILE = "datasets/code_alpaca_500_train.json"
EVOLVED_FILE = "datasets/code_alpaca_100_evolved.json"
OUTPUT_FILE = "datasets/code_alpaca_200_evol_train.json"

with open(ORIGINAL_FILE, encoding="utf-8") as f:
    original = json.load(f)[:100]

with open(EVOLVED_FILE, encoding="utf-8") as f:
    evolved = json.load(f)

assert len(original) == 100
assert len(evolved) == 100

dataset = []

# 100 original samples
for x in original:
    dataset.append({
        "instruction": x["instruction"],
        "input": x.get("input", ""),
        "output": x["output"]
    })

# 100 evolved samples
for x in evolved:
    dataset.append({
        "instruction": x["enhanced_instruction"],
        "input": x.get("original_input", ""),
        "output": x["evolved_output"]
    })

assert len(dataset) == 200

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("=" * 70)
print("CODE-ALPACA 200 EVOLVED DATASET CREATED")
print("=" * 70)
print("Original samples :", 100)
print("Evolved samples  :", 100)
print("Total samples    :", len(dataset))
print("Saved            :", OUTPUT_FILE)
