import argparse
import gzip
import json
import signal
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_LORA_PATH = "/workspace/project/outputs/qwen-codeevol-200-lora"
DEFAULT_PREDICTIONS = "/workspace/project/results/codeevol_200_scot_humaneval_predictions.jsonl"
DEFAULT_OUT_DIR = "/workspace/project/results/self_debug"
LOCAL_HUMANEVAL_PATH = Path("/workspace/project/datasets/humaneval/HumanEval.jsonl.gz")

MAX_NEW_TOKENS = 512
EXEC_TIMEOUT_SEC = 10

SIMPLE_DEBUG_SYSTEM = """You are an expert competitive programmer debugging a failed Python solution.
Analyze your previous Structured Chain of Thought (SCoT) and Python implementation carefully to identify logical flaws, missing imports, or unhandled edge cases.
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


def evaluate_task(code: str, problem: dict, timeout_sec: int = EXEC_TIMEOUT_SEC) -> tuple[bool, str]:
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


def generate_chat(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Strategy 1: Simple Feedback Self-Debugging on HumanEval")
    parser.add_argument("--lora-path", default=DEFAULT_LORA_PATH)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-iters", type=int, default=1, help="Number of Simple Feedback debug rounds")
    parser.add_argument("--gpu", type=int, default=0, help="Target CUDA GPU index")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_out_path = out_dir / "s1_simple_humaneval_predictions.jsonl"
    res_out_path = out_dir / "s1_simple_humaneval_results.jsonl"

    print("=" * 70, flush=True)
    print("STRATEGY 1 (SIMPLE FEEDBACK) SELF-DEBUGGING - HUMANEVAL", flush=True)
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
                messages = [
                    {"role": "system", "content": SIMPLE_DEBUG_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Programming Problem:\n\n{problem['prompt']}\n\n"
                            f"Structured Chain of Thought:\n\n{init_scot}\n\n"
                            f"Previous Python Solution:\n\n{current_code}\n\n"
                            "Feedback: The above Python code is incorrect. "
                            "Please review the problem and your reasoning carefully, and generate the corrected Python solution."
                        ),
                    },
                ]
                raw_out = generate_chat(model, tokenizer, messages, max_new_tokens=MAX_NEW_TOKENS)
                candidate_code = clean_code(raw_out)

                if "def " in candidate_code and not candidate_code.lstrip().startswith(("def ", "import ", "from ")):
                    candidate_code = candidate_code[candidate_code.find("def ") :].strip()

                current_code = candidate_code
                passed, status = evaluate_task(current_code, problem)
                debug_steps.append({
                    "step": step,
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
    print("STRATEGY 1 (SIMPLE FEEDBACK) SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Initial Passed : {initial_passed}/{total} ({initial_passed / total * 100:.2f}%)", flush=True)
    print(f"Final Passed   : {final_passed}/{total} ({final_passed / total * 100:.2f}%)", flush=True)
    print(f"Net Gain       : +{final_passed - initial_passed} tasks (+{(final_passed - initial_passed) / total * 100:.2f} pp)", flush=True)
    print(f"Predictions    : {pred_out_path}", flush=True)
    print(f"Results        : {res_out_path}", flush=True)


if __name__ == "__main__":
    main()
