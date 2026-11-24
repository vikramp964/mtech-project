import json
import traceback
from datasets import load_dataset

PREDICTIONS = "/workspace/project/results/self_scot_humaneval_predictions.jsonl"
OUTPUT = "/workspace/project/results/self_scot_humaneval_results.jsonl"

dataset = load_dataset("openai/openai_humaneval")["test"]
problems = {x["task_id"]: x for x in dataset}

def clean_code(code):
    code = code.strip()

    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    elif code.startswith("```"):
        code = code[3:].strip()

    if code.endswith("```"):
        code = code[:-3].strip()

    return code

results = []
passed = 0
total = 0

with open(PREDICTIONS, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        item = json.loads(line)
        task_id = item["task_id"]
        problem = problems[task_id]

        code = clean_code(item["completion"])

        namespace = {}

        try:
            # Execute generated solution
            exec(code, namespace)

            # Execute HumanEval test code
            exec(problem["test"], namespace)

            # Run tests
            check = namespace["check"]
            candidate = namespace[problem["entry_point"]]

            check(candidate)

            result = "passed"
            passed += 1

        except Exception as e:
            result = f"failed: {type(e).__name__}: {e}"

        total += 1

        results.append({
            "task_id": task_id,
            "passed": result == "passed",
            "result": result,
        })

        print(f"{task_id}: {result}")

with open(OUTPUT, "w", encoding="utf-8") as f:
    for x in results:
        f.write(json.dumps(x) + "\n")

print("\n" + "=" * 70)
print("SELF-SCoT HUMAN EVAL RESULTS")
print("=" * 70)
print(f"Passed : {passed}/{total}")
print(f"Pass@1 : {passed/total*100:.2f}%")
print(f"Results: {OUTPUT}")
