import json
from pathlib import Path

PROBLEMS = "external/mxeval/data/mbxp/mbcpp_release_v1.2.jsonl"
OFFICIAL = "external/mxeval/data/mbxp/examples/mbcpp_samples.jsonl"

BASELINE = "results/codeevol_200_mbcpp_predictions.jsonl_results.jsonl"
SCOT = "results/codeevol_200_scot_mbcpp_predictions.jsonl_results.jsonl"

def load(path):
    with open(path, encoding="utf-8") as f:
        return {
            x["task_id"]: x
            for x in (json.loads(line) for line in f if line.strip())
        }

problems = load(PROBLEMS)
official = load(OFFICIAL)
baseline = load(BASELINE)
scot = load(SCOT)

print("=" * 80)
print("MBCPP CONTROLLED DIAGNOSTIC")
print("=" * 80)

print("Problems :", len(problems))
print("Official :", len(official))
print("Baseline :", len(baseline))
print("SCoT     :", len(scot))

# Select tasks where our systems fail.
failed = [
    tid for tid in problems
    if tid in baseline
    and tid in scot
    and not baseline[tid]["passed"]
    and not scot[tid]["passed"]
]

# Also include some cases where baseline succeeds.
passed_baseline = [
    tid for tid in problems
    if tid in baseline
    and baseline[tid]["passed"]
]

selected = failed[:10] + passed_baseline[:10]

# Remove duplicates while preserving order.
selected = list(dict.fromkeys(selected))

print("\nSelected tasks:", selected)

out = Path("results/mbcpp_diagnostic.txt")

with out.open("w", encoding="utf-8") as f:

    for num, tid in enumerate(selected, 1):

        p = problems[tid]
        o = official.get(tid, {})
        b = baseline.get(tid, {})
        s = scot.get(tid, {})

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"CASE {num}: {tid}\n")
        f.write("=" * 80 + "\n")

        f.write("\n--- PROBLEM ---\n")
        f.write(p.get("prompt", "") + "\n")

        f.write("\n--- ENTRY POINT ---\n")
        f.write(str(p.get("entry_point", "")) + "\n")

        f.write("\n--- LANGUAGE ---\n")
        f.write(str(p.get("language", "")) + "\n")

        f.write("\n--- OFFICIAL COMPLETION ---\n")
        f.write(o.get("completion", "[NOT FOUND]") + "\n")

        f.write("\n--- BASELINE COMPLETION ---\n")
        f.write(b.get("completion", "[NOT FOUND]") + "\n")

        f.write("\n--- BASELINE RESULT ---\n")
        f.write("PASSED: " + str(b.get("passed")) + "\n")
        f.write("RESULT: " + repr(b.get("result", "")) + "\n")

        f.write("\n--- SCoT REASONING ---\n")
        f.write(s.get("scot", "[NOT FOUND]") + "\n")

        f.write("\n--- SCoT COMPLETION ---\n")
        f.write(s.get("completion", "[NOT FOUND]") + "\n")

        f.write("\n--- SCoT RESULT ---\n")
        f.write("PASSED: " + str(s.get("passed")) + "\n")
        f.write("RESULT: " + repr(s.get("result", "")) + "\n")

print("\nDiagnostic written to:")
print(out)
