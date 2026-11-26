import argparse
import ast
import doctest
import gzip
import json
import re
import signal
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_LORA_PATH = "/workspace/project/outputs/qwen-codeevol-200-lora"
DEFAULT_PREDICTIONS = "/workspace/project/results/Phase_2/codeevol_200_scot_humaneval_predictions.jsonl"
FALLBACK_PREDICTIONS = "/workspace/project/results/codeevol_200_scot_humaneval_predictions.jsonl"
DEFAULT_OUT_DIR = "/workspace/project/results/phase3/unit_test_explanation"
LOCAL_HUMANEVAL_PATH = Path("/workspace/project/datasets/humaneval/HumanEval.jsonl.gz")

EXPLAIN_MAX_TOKENS = 512
REPAIR_MAX_TOKENS = 512
EXEC_TIMEOUT_SEC = 10

EXPLAIN_SYSTEM = """You are an expert Python debugging assistant performing fault localization on a solution known to be incorrect.
Given the problem requirements, the previous Python code (known to be wrong), and the public unit-test feedback:
1. Explain step-by-step what the old code currently does.
2. Identify the exact bug and possible chances of failure (edge cases like empty/negative inputs, boundary conditions, indexing, types, or missing imports).
Do NOT write the corrected Python code yet."""

REPAIR_SYSTEM = """You are an expert competitive programmer fixing a Python solution known to be incorrect.
Use the problem requirements, previous code, public unit-test feedback, and the old code's explanation and failure analysis to produce a corrected solution.
Return ONLY the corrected Python code. Do not explain the solution. Do not use markdown code fences."""


class _ExecTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _ExecTimeout("Execution timed out")


def clean_code(code: str) -> str:
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python") :].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[: -3].strip()
    return code


def extract_docstring_tests(prompt: str, entry_point: str) -> list[dict]:
    """Extract visible/public examples from the function docstring in problem['prompt']."""
    docstring = ""
    try:
        tree = ast.parse(prompt)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                docstring = ast.get_docstring(node) or ""
                break
    except SyntaxError:
        docstring = prompt

    if not docstring:
        return []

    tests = []
    parser = doctest.DocTestParser()
    for ex in parser.get_examples(docstring):
        src = ex.source.strip()
        want = ex.want.strip()
        if src and want:
            tests.append({"type": "doctest", "call": src, "expected_str": want})

    if not tests:
        pattern = re.compile(rf"({re.escape(entry_point)}\s*\(.*?\))\s*(?:==|=>|->)\s*(.+)")
        for line in docstring.splitlines():
            m = pattern.search(line.strip())
            if m:
                tests.append({
                    "type": "expr",
                    "call": m.group(1).strip(),
                    "expected_str": m.group(2).strip().rstrip("."),
                })

    return tests


def run_public_unit_tests(code: str, problem: dict, timeout_sec: int = EXEC_TIMEOUT_SEC) -> tuple[bool, str, int]:
    """Execute visible docstring tests and return (all_public_passed, feedback_message, num_public_tests)."""
    tests = extract_docstring_tests(problem["prompt"], problem["entry_point"])
    if not tests:
        return (
            False,
            "No runnable docstring examples were found, but your solution is known to be incorrect. "
            "Check problem requirements, possible failure causes (edge cases, empty/negative inputs, bounds), and missing imports.",
            0,
        )

    namespace = {}
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_sec)
    try:
        exec(clean_code(code), namespace)
        for t in tests:
            call_src = t["call"]
            exp_str = t["expected_str"]
            try:
                actual_val = eval(call_src, namespace)
            except Exception as e:
                return (
                    False,
                    f"Public unit test failed with runtime error:\n"
                    f"  Test Input : {call_src}\n"
                    f"  Expected   : {exp_str}\n"
                    f"  Actual     : Raised {type(e).__name__}: {e}",
                    len(tests),
                )

            try:
                expected_val = eval(exp_str, namespace)
                matched = actual_val == expected_val
            except Exception:
                matched = repr(actual_val).strip() == exp_str.strip()

            if not matched:
                return (
                    False,
                    f"Public unit test failed with wrong output:\n"
                    f"  Test Input : {call_src}\n"
                    f"  Expected   : {exp_str}\n"
                    f"  Actual     : {repr(actual_val)}",
                    len(tests),
                )

        return (
            True,
            f"All {len(tests)} public docstring example(s) passed, but your solution is still incorrect on hidden tests. "
            "Check possible failure causes: boundary/edge cases (empty inputs, negative numbers, duplicates, ordering) or missing imports.",
            len(tests),
        )
    except Exception as e:
        return (
            False,
            f"Code failed during compilation/definition:\n  Error: {type(e).__name__}: {e}",
            len(tests),
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def evaluate_task(code: str, problem: dict, timeout_sec: int = EXEC_TIMEOUT_SEC) -> tuple[bool, str]:
    """Final grading against held-out problem['test']."""
    namespace = {}
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_sec)
    try:
        exec(clean_code(code), namespace)
        exec(problem["test"], namespace)
        check = namespace["check"]
        candidate = namespace[problem["entry_point"]]
        check(candidate)
        return True, "passed"
    except Exception as e:
        return False, f"failed: {type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def load_humaneval_problems() -> dict[str, dict]:
    if LOCAL_HUMANEVAL_PATH.exists():
        problems = {}
        with gzip.open(LOCAL_HUMANEVAL_PATH, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    problems[row["task_id"]] = row
        return problems
    dataset = load_dataset("openai/openai_humaneval")["test"]
    return {row["task_id"]: row for row in dataset}


def generate_chat(model, tokenizer, messages: list[dict], max_new_tokens: int) -> tuple[str, dict]:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    max_ctx = getattr(model.config, "max_position_embeddings", 32768)
    avail_new = max_ctx - input_len - 16
    if avail_new <= 0:
        # Trim oldest user context if prompt ever approaches context ceiling
        max_keep_tokens = max(256, max_ctx - max_new_tokens - 32)
        inputs["input_ids"] = inputs["input_ids"][:, -max_keep_tokens:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -max_keep_tokens:]
        input_len = inputs["input_ids"].shape[-1]
        avail_new = max_ctx - input_len - 16

    effective_max_new = max(1, min(max_new_tokens, avail_new))
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=effective_max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    meta = {
        "prompt_tokens": int(input_len),
        "generated_tokens": int(gen_ids.shape[-1]),
        "hit_max_new_tokens": bool(gen_ids.shape[-1] >= effective_max_new),
    }
    return text, meta


def main():
    default_pred = DEFAULT_PREDICTIONS if Path(DEFAULT_PREDICTIONS).exists() else FALLBACK_PREDICTIONS
    parser = argparse.ArgumentParser(description="Strategy 3: Public Unit-Test Feedback + Explanation Self-Debugging on HumanEval")
    parser.add_argument("--lora-path", default=DEFAULT_LORA_PATH)
    parser.add_argument("--predictions", default=default_pred)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-iters", type=int, default=1, help="Number of Explanation-Guided debug rounds")
    parser.add_argument("--gpu", type=int, default=0, help="Target CUDA GPU index")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_path = out_dir / "s3_explanation_humaneval_predictions.jsonl"
    res_out_path = out_dir / "s3_explanation_humaneval_results.jsonl"

    print("=" * 70, flush=True)
    print("STRATEGY 3 (UNIT-TEST FEEDBACK + EXPLANATION) SELF-DEBUGGING - HUMANEVAL", flush=True)
    print("=" * 70, flush=True)

    problems = load_humaneval_problems()

    initial_preds = []
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                initial_preds.append(json.loads(line))

    initial_eval = {}
    initial_passed = 0
    for item in initial_preds:
        tid = item["task_id"]
        passed, status = evaluate_task(item["completion"], problems[tid])
        initial_eval[tid] = {"passed": passed, "result": status}
        if passed:
            initial_passed += 1

    failed_count = len(initial_preds) - initial_passed
    print(f"Initial SCoT Pass@1 : {initial_passed}/{len(initial_preds)} ({initial_passed / len(initial_preds) * 100:.2f}%)", flush=True)
    print(f"Failed Tasks Targeted: {failed_count}", flush=True)

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading Qwen2.5-Coder-7B-Instruct + LoRA onto {device_str}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": args.gpu} if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()
    print("Model loaded.\n", flush=True)

    final_records = []
    final_results = []
    final_passed = 0

    for idx, item in enumerate(initial_preds):
        tid = item["task_id"]
        problem = problems[tid]
        init_scot = item.get("scot", "")
        init_code = clean_code(item["completion"])
        passed = initial_eval[tid]["passed"]
        status = initial_eval[tid]["result"]
        current_code = init_code
        debug_steps = []

        if not passed:
            for step in range(1, args.max_iters + 1):
                pub_ok, feedback_msg, num_pub = run_public_unit_tests(current_code, problem)

                # Stage A: Old code explanation & possible chances of failure
                explain_messages = [
                    {"role": "system", "content": EXPLAIN_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Problem Requirements:\n\n{problem['prompt']}\n\n"
                            f"Old Code (Known to be Incorrect):\n\n{current_code}\n\n"
                            f"Unit-Test Feedback:\n{feedback_msg}\n\n"
                            "Explain what the old code currently does and list the bug and possible chances of failure."
                        ),
                    },
                ]
                code_explanation, exp_meta = generate_chat(
                    model, tokenizer, explain_messages, max_new_tokens=EXPLAIN_MAX_TOKENS
                )

                # Stage B: Generate new code from requirements, unit-test feedback, and old code explanation
                repair_messages = [
                    {"role": "system", "content": REPAIR_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Problem Requirements:\n\n{problem['prompt']}\n\n"
                            f"Structured Chain of Thought:\n\n{init_scot}\n\n"
                            f"Old Code (Incorrect):\n\n{current_code}\n\n"
                            f"Unit-Test Feedback:\n{feedback_msg}\n\n"
                            f"Old Code Explanation & Possible Chances of Failure:\n{code_explanation}\n\n"
                            "Generate the corrected Python code now."
                        ),
                    },
                ]
                raw_out, rep_meta = generate_chat(
                    model, tokenizer, repair_messages, max_new_tokens=REPAIR_MAX_TOKENS
                )
                candidate_code = clean_code(raw_out)

                if "def " in candidate_code and not candidate_code.lstrip().startswith(("def ", "import ", "from ")):
                    candidate_code = candidate_code[candidate_code.find("def ") :].strip()

                current_code = candidate_code
                passed, status = evaluate_task(current_code, problem)
                debug_steps.append({
                    "step": step,
                    "public_tests_count": num_pub,
                    "public_tests_passed_before_fix": pub_ok,
                    "feedback": feedback_msg,
                    "explanation": code_explanation,
                    "explain_tokens": exp_meta,
                    "repair_tokens": rep_meta,
                    "completion": current_code,
                    "passed": passed,
                    "result": status,
                })
                if passed:
                    break

            print(f"[{idx + 1}/{len(initial_preds)}] {tid} (debugged): {initial_eval[tid]['result']} -> {status}", flush=True)
        else:
            print(f"[{idx + 1}/{len(initial_preds)}] {tid} (kept): passed", flush=True)

        if passed:
            final_passed += 1

        final_records.append({
            "task_id": tid,
            "prompt": item["prompt"],
            "scot": init_scot,
            "initial_completion": init_code,
            "initial_passed": initial_eval[tid]["passed"],
            "initial_result": initial_eval[tid]["result"],
            "completion": current_code,
            "passed": passed,
            "result": status,
            "debug_steps": debug_steps,
        })

        final_results.append({
            "task_id": tid,
            "initial_passed": initial_eval[tid]["passed"],
            "passed": passed,
            "result": status,
        })

    with open(pred_out_path, "w", encoding="utf-8") as f:
        for rec in final_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(res_out_path, "w", encoding="utf-8") as f:
        for res in final_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    total = len(final_records)
    print("\n" + "=" * 70, flush=True)
    print("STRATEGY 3 (UNIT-TEST FEEDBACK + EXPLANATION) SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Initial Passed : {initial_passed}/{total} ({initial_passed / total * 100:.2f}%)", flush=True)
    print(f"Final Passed   : {final_passed}/{total} ({final_passed / total * 100:.2f}%)", flush=True)
    print(f"Net Gain       : +{final_passed - initial_passed} tasks (+{(final_passed - initial_passed) / total * 100:.2f} pp)", flush=True)
    print(f"Predictions    : {pred_out_path}", flush=True)
    print(f"Results        : {res_out_path}", flush=True)


if __name__ == "__main__":
    main()
