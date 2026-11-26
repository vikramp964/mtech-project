import json
import shutil
import subprocess

RESULTS_DIR = "/workspace/project/results"

input_file = f"{RESULTS_DIR}/codealpaca_500_pass2_predictions.jsonl"
output_file = f"{RESULTS_DIR}/codealpaca_500_pass2_results.jsonl"
temp_file = "/tmp/codealpaca_500_pass2_expanded.jsonl"

with open(input_file, encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

print("=" * 70)
print("CODE-ALPACA-500 PASS@2 FUNCTIONAL EVALUATION")
print("=" * 70)

print(f"Problems: {len(data)}")

candidate_count = sum(len(x["completions"]) for x in data)
print(f"Candidates: {candidate_count}")

with open(temp_file, "w", encoding="utf-8") as f:
    for item in data:
        for completion in item["completions"]:
            f.write(json.dumps({
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "completion": completion
            }) + "\n")

print(f"Expanded candidates: {candidate_count}")

subprocess.run([
    "python",
    "-m",
    "human_eval.evaluate_functional_correctness",
    temp_file,
    "--problem_file",
    "datasets/humaneval/HumanEval.jsonl.gz"
], check=True)

generated_results = temp_file + "_results.jsonl"

shutil.copyfile(generated_results, output_file)

print("\n" + "=" * 70)
print("CODE-ALPACA-500 PASS@2 FUNCTIONAL EVALUATION COMPLETE")
print("=" * 70)
print(f"Saved: {output_file}")
