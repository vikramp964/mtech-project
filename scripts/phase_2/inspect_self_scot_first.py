import json

path = "results/self_scot_humaneval_predictions.jsonl"

# ============================================================
# 1. Read the first line exactly as stored
# ============================================================

with open(path, encoding="utf-8") as f:
    first_line = f.readline().rstrip("\n")

print("=" * 80)
print("1. RAW FIRST JSONL LINE")
print("=" * 80)
print(first_line)

# ============================================================
# 2. Parse JSON
# ============================================================

record = json.loads(first_line)

print("\n" + "=" * 80)
print("2. PRETTY-FORMATTED JSON")
print("=" * 80)
print(json.dumps(record, indent=2, ensure_ascii=False))

# ============================================================
# 3. Individual components
# ============================================================

print("\n" + "=" * 80)
print("3. TASK ID")
print("=" * 80)
print(record["task_id"])

print("\n" + "=" * 80)
print("4. ORIGINAL HUMAN EVAL PROMPT")
print("=" * 80)
print(record["prompt"])

print("\n" + "=" * 80)
print("5. GENERATED SELF-SCoT")
print("=" * 80)
print(record["scot"])

print("\n" + "=" * 80)
print("6. GENERATED CODE")
print("=" * 80)
print(record["completion"])

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
