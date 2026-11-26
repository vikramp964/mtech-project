import json
from pathlib import Path

PREDICTIONS_PATH = Path("/workspace/project/results/phase3/unit_test_trace/s4_trace_humaneval_predictions.jsonl")
ANALYSIS_DIR = Path("/workspace/project/results/phase3/unit_test_trace/s4_trace_analysis")

SCOT_REGRESSIONS = {"HumanEval/26", "HumanEval/54", "HumanEval/56", "HumanEval/116"}


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    with open(PREDICTIONS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    w2r, w2w, r2r, r2w = [], [], [], []
    recovered_regressions = []
    truncated_calls = 0

    for rec in records:
        before = rec["initial_passed"]
        after = rec["passed"]
        steps = rec.get("debug_steps", [])
        for st in steps:
            if st.get("repair_tokens", {}).get("hit_max_new_tokens"):
                truncated_calls += 1

        entry = {
            "task_id": rec["task_id"],
            "was_scot_regression": rec["task_id"] in SCOT_REGRESSIONS,
            "before_passed": before,
            "after_passed": after,
            "before_result": rec["initial_result"],
            "after_result": rec["result"],
            "debug_steps": steps,
            "scot_reasoning": rec["scot"],
            "before_completion": rec["initial_completion"],
            "after_completion": rec["completion"],
            "problem_prompt": rec["prompt"],
        }

        if not before and after:
            w2r.append(entry)
            if rec["task_id"] in SCOT_REGRESSIONS:
                recovered_regressions.append(rec["task_id"])
        elif not before and not after:
            w2w.append(entry)
        elif before and after:
            r2r.append(entry)
        else:
            r2w.append(entry)

    total = len(records)
    before_pass = len(r2r) + len(r2w)
    after_pass = len(r2r) + len(w2r)

    summary = {
        "strategy": "Strategy 4: Public Unit-Test Feedback + Execution Trace (E5)",
        "total_tasks": total,
        "before_scot_pass": before_pass,
        "before_scot_pass_rate": round(before_pass / total * 100, 2),
        "after_s4_pass": after_pass,
        "after_s4_pass_rate": round(after_pass / total * 100, 2),
        "net_gain_tasks": after_pass - before_pass,
        "net_gain_pp": round((after_pass - before_pass) / total * 100, 2),
        "truncated_generation_calls": truncated_calls,
        "transitions": {
            "wrong_to_right": len(w2r),
            "wrong_to_wrong": len(w2w),
            "right_to_right": len(r2r),
            "right_to_wrong": len(r2w),
        },
        "recovered_scot_regressions": recovered_regressions,
    }

    with open(ANALYSIS_DIR / "transition_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for name, items in [
        ("wrong_to_right_all.json", w2r),
        ("wrong_to_wrong_all.json", w2w),
        ("right_to_right_all.json", r2r),
        ("right_to_wrong_all.json", r2w),
    ]:
        with open(ANALYSIS_DIR / name, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)

    lines = [
        "### ============================================================ ❌ → ✅  RECOVERED BY STRATEGY 4 (UNIT-TEST FEEDBACK + EXECUTION TRACE)",
        " ".join(x["task_id"] for x in w2r) or "None",
        "",
        "### ============================================================ RECOVERED PHASE-2 SCoT REGRESSIONS",
        " ".join(recovered_regressions) or "None",
        "",
        "### ============================================================ ❌ → ❌  REMAINED WRONG",
        " ".join(x["task_id"] for x in w2w) or "None",
        "",
        "### ============================================================ ✅ → ✅  REMAINED CORRECT",
        " ".join(x["task_id"] for x in r2r) or "None",
        "",
        "### ============================================================ ✅ → ❌  NEW REGRESSIONS",
        " ".join(x["task_id"] for x in r2w) or "None",
    ]
    with open(ANALYSIS_DIR / "transition_cases.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))
    print("\n" + "\n".join(lines[:10]))


if __name__ == "__main__":
    main()
