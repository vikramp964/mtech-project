import argparse
import ast
import doctest
import gzip
import json
import re
import signal
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_LORA_PATH = "/workspace/project/outputs/qwen-codeevol-200-lora"
DEFAULT_PREDICTIONS = "/workspace/project/results/Phase_2/codeevol_200_scot_humaneval_predictions.jsonl"
FALLBACK_PREDICTIONS = "/workspace/project/results/codeevol_200_scot_humaneval_predictions.jsonl"
DEFAULT_OUT_DIR = "/workspace/project/results/phase3/unit_test_trace"
LOCAL_HUMANEVAL_PATH = Path("/workspace/project/datasets/humaneval/HumanEval.jsonl.gz")

REPAIR_MAX_TOKENS = 640
EXEC_TIMEOUT_SEC = 10
MAX_TRACE_STEPS = 35
MAX_VAR_REPR_LEN = 80

TRACE_DEBUG_SYSTEM = """You are an expert competitive programmer debugging a failed Python solution using unit-test feedback and a step-by-step execution trace.
Inspect the problem requirements, your Structured Chain of Thought (SCoT), the previous Python implementation, the unit-test feedback, and the runtime execution trace carefully to spot where variable values, loop bounds, branches, or edge cases diverge from expected behavior.
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


def _compact_repr(val, max_len: int = MAX_VAR_REPR_LEN) -> str:
    try:
        s = repr(val)
    except Exception as e:
        s = f"<unreprable {type(e).__name__}>"
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _format_locals(f_locals: dict) -> str:
    items = []
    for k, v in f_locals.items():
        if k.startswith("__"):
            continue
        if callable(v) or hasattr(v, "__spec__"):
            continue
        items.append(f"{k}={_compact_repr(v)}")
    return ", ".join(items) if items else "{}"


def capture_execution_trace(
    code: str,
    call_expr: str,
    expected_str: str,
    timeout_sec: int = EXEC_TIMEOUT_SEC,
    max_steps: int = MAX_TRACE_STEPS,
) -> dict:
    """Run call_expr against cleaned code with sys.settrace to capture a bounded step-by-step execution trace."""
    cleaned = clean_code(code)
    source_lines = cleaned.splitlines()
    events = []
    total_steps = 0
    outcome = ""

    def tracer(frame, event, arg):
        nonlocal total_steps
        if frame.f_code.co_filename != "<string>":
            return tracer
        if event == "line":
            total_steps += 1
            lineno = frame.f_lineno
            src_line = source_lines[lineno - 1].strip() if 1 <= lineno <= len(source_lines) else ""
            events.append({
                "step": total_steps,
                "line": lineno,
                "code": src_line,
                "locals": _format_locals(frame.f_locals),
            })
        elif event == "return":
            total_steps += 1
            events.append({
                "step": total_steps,
                "line": frame.f_lineno,
                "code": f"[return -> {_compact_repr(arg)}]",
                "locals": _format_locals(frame.f_locals),
            })
        elif event == "exception":
            total_steps += 1
            exc_type, exc_val, _ = arg
            exc_name = exc_type.__name__ if exc_type else "Exception"
            events.append({
                "step": total_steps,
                "line": frame.f_lineno,
                "code": f"[exception -> {exc_name}: {_compact_repr(exc_val, 60)}]",
                "locals": _format_locals(frame.f_locals),
            })
        return tracer

    namespace = {}
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_sec)
    try:
        exec(compile(cleaned, "<string>", "exec"), namespace)
        sys.settrace(tracer)
        try:
            actual_val = eval(compile(call_expr, "<trace_call>", "eval"), namespace)
            outcome = f"Returned {_compact_repr(actual_val)} (Expected: {expected_str})"
        except Exception as e:
            outcome = f"Raised {type(e).__name__}: {e} (Expected: {expected_str})"
    except Exception as e:
        outcome = f"Failed before/during execution: {type(e).__name__}: {e}"
    finally:
        sys.settrace(None)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    omitted = 0
    if len(events) > max_steps:
        head_n = max_steps // 2
        tail_n = max_steps - head_n
        omitted = len(events) - (head_n + tail_n)
        kept_events = events[:head_n] + [{"omitted": omitted}] + events[-tail_n:]
    else:
        kept_events = events

    lines = [f"Traced Call: {call_expr}", f"Outcome    : {outcome}"]
    if not kept_events:
        lines.append("  (No function line steps captured)")
    else:
        for ev in kept_events:
            if "omitted" in ev:
                lines.append(f"  ... ({ev['omitted']} intermediate steps omitted) ...")
            else:
                lines.append(
                    f"  Step {ev['step']:>2} | L{ev['line']:>2}: {ev['code']} | locals: {ev['locals']}"
                )

    return {
        "traced_call": call_expr,
        "outcome": outcome,
        "total_captured_steps": total_steps,
        "omitted_steps": omitted,
        "trace_text": "\n".join(lines),
    }


def run_public_unit_tests_with_trace(
    code: str, problem: dict, timeout_sec: int = EXEC_TIMEOUT_SEC
) -> tuple[bool, str, int, dict]:
    """Execute visible docstring tests and capture an execution trace on the failing (or first public) test."""
    tests = extract_docstring_tests(problem["prompt"], problem["entry_point"])
    if not tests:
        trace_info = capture_execution_trace(
            code, f"{problem['entry_point']}()", "<hidden test>", timeout_sec=timeout_sec
        )
        return (
            False,
            "No runnable docstring examples were found, but your solution is known to be incorrect. "
            "Check problem requirements, possible failure causes (edge cases, empty/negative inputs, bounds), and missing imports.",
            0,
            trace_info,
        )

    namespace = {}
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_sec)
    failing_test = None
    feedback_msg = ""
    all_passed = False

    try:
        exec(clean_code(code), namespace)
        for t in tests:
            call_src = t["call"]
            exp_str = t["expected_str"]
            try:
                actual_val = eval(call_src, namespace)
            except Exception as e:
                failing_test = t
                feedback_msg = (
                    f"Public unit test failed with runtime error:\n"
                    f"  Test Input : {call_src}\n"
                    f"  Expected   : {exp_str}\n"
                    f"  Actual     : Raised {type(e).__name__}: {e}"
                )
                break

            try:
                expected_val = eval(exp_str, namespace)
                matched = actual_val == expected_val
            except Exception:
                matched = repr(actual_val).strip() == exp_str.strip()

            if not matched:
                failing_test = t
                feedback_msg = (
                    f"Public unit test failed with wrong output:\n"
                    f"  Test Input : {call_src}\n"
                    f"  Expected   : {exp_str}\n"
                    f"  Actual     : {repr(actual_val)}"
                )
                break

        if failing_test is None:
            all_passed = True
            feedback_msg = (
                f"All {len(tests)} public docstring example(s) passed, but your solution failed on hidden edge-case tests. "
                "Check boundary conditions (empty inputs, negative numbers, duplicates, ordering, or missing imports)."
            )
    except Exception as e:
        failing_test = tests[0]
        feedback_msg = f"Code failed during compilation/definition:\n  Error: {type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    target_test = failing_test if failing_test is not None else tests[0]
    trace_info = capture_execution_trace(
        code, target_test["call"], target_test["expected_str"], timeout_sec=timeout_sec
    )
    return all_passed, feedback_msg, len(tests), trace_info


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
    parser = argparse.ArgumentParser(description="Strategy 4: Public Unit-Test Feedback + Execution Trace Self-Debugging on HumanEval")
    parser.add_argument("--lora-path", default=DEFAULT_LORA_PATH)
    parser.add_argument("--predictions", default=default_pred)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-iters", type=int, default=1, help="Number of Trace-Guided debug rounds")
    parser.add_argument("--gpu", type=int, default=0, help="Target CUDA GPU index")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_path = out_dir / "s4_trace_humaneval_predictions.jsonl"
    res_out_path = out_dir / "s4_trace_humaneval_results.jsonl"

    print("=" * 70, flush=True)
    print("STRATEGY 4 (UNIT-TEST FEEDBACK + EXECUTION TRACE) SELF-DEBUGGING - HUMANEVAL", flush=True)
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
                pub_ok, feedback_msg, num_pub, trace_info = run_public_unit_tests_with_trace(
                    current_code, problem
                )

                repair_messages = [
                    {"role": "system", "content": TRACE_DEBUG_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Programming Problem:\n\n{problem['prompt']}\n\n"
                            f"Structured Chain of Thought:\n\n{init_scot}\n\n"
                            f"Previous Python Solution:\n\n{current_code}\n\n"
                            f"Unit-Test Feedback:\n{feedback_msg}\n\n"
                            f"Runtime Execution Trace:\n{trace_info['trace_text']}\n\n"
                            "Using the execution trace and unit-test feedback, fix the bug and generate the corrected Python solution."
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
                    "trace": {
                        "traced_call": trace_info["traced_call"],
                        "outcome": trace_info["outcome"],
                        "total_captured_steps": trace_info["total_captured_steps"],
                        "omitted_steps": trace_info["omitted_steps"],
                    },
                    "trace_text": trace_info["trace_text"],
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
    print("STRATEGY 4 (UNIT-TEST FEEDBACK + EXECUTION TRACE) SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Initial Passed : {initial_passed}/{total} ({initial_passed / total * 100:.2f}%)", flush=True)
    print(f"Final Passed   : {final_passed}/{total} ({final_passed / total * 100:.2f}%)", flush=True)
    print(f"Net Gain       : +{final_passed - initial_passed} tasks (+{(final_passed - initial_passed) / total * 100:.2f} pp)", flush=True)
    print(f"Predictions    : {pred_out_path}", flush=True)
    print(f"Results        : {res_out_path}", flush=True)


if __name__ == "__main__":
    main()
