import json
import shutil
import os
import subprocess

RESULTS_DIR = "/workspace/project/results"

MODELS = [
    "base",
    "alpaca",
    "evol_scot",
]

for model in MODELS:

    input_file = f"{RESULTS_DIR}/{model}_pass3_predictions.jsonl"
    output_file = f"{RESULTS_DIR}/{model}_pass3_results.jsonl"

    print("\n" + "=" * 70)
    print(f"Evaluating: {model}")
    print("=" * 70)

    # Create temporary HumanEval-style prediction file
    temp_file = f"/tmp/{model}_pass3_expanded.jsonl"

    with open(input_file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    with open(temp_file, "w", encoding="utf-8") as f:

        for item in data:

            for completion in item["completions"]:

                record = {
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "completion": completion,
                }

                f.write(json.dumps(record) + "\n")

    print(f"Expanded candidates: {len(data) * 3}")

    # Run HumanEval evaluator
    cmd = [
        "python",
        "-m",
        "human_eval.evaluate_functional_correctness",
        temp_file,
        "--problem_file",
        "datasets/humaneval/HumanEval.jsonl.gz",
    ]

    subprocess.run(cmd, check=True)

    # HumanEval creates <file>_results.jsonl
    generated_results = temp_file + "_results.jsonl"

    # Move result file
    shutil.copyfile(
        generated_results,
        output_file,
    )

    print(f"Saved: {output_file}")

print("\nPASS@2 FUNCTIONAL EVALUATION COMPLETE")
