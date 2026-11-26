import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "/workspace/project/outputs/qwen-codeevol-200-lora"

INPUT_FILE = "/workspace/project/external/mxeval/data/mbxp/mbcpp_release_v1.2.jsonl"
OUTPUT_FILE = "/workspace/project/results/codeevol_200_scot_mbcpp_predictions.jsonl"

MAX_SCOT_TOKENS = 512
MAX_CODE_TOKENS = 768

TEMPERATURE = 0.7
NUM_SAMPLES = 1


def clean_completion(text, entry_point):
    text = text.strip()

    # Remove markdown fences
    text = re.sub(
        r"^\s*```(?:cpp|c\+\+)?\s*",
        "",
        text,
        flags=re.I
    )
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    # Detect function declaration
    pattern = (
        rf"\b{re.escape(entry_point)}\s*"
        r"\([^{}]*\)\s*\{"
    )

    match = re.search(pattern, text, flags=re.S)

    if not match:
        return text

    # Find opening brace
    open_brace = text.find("{", match.start())
    depth = 0

    # Extract balanced function body
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1

            if depth == 0:
                return text[open_brace + 1:i].strip()

    # Function was truncated before closing brace.
    # Return everything after the opening brace.
    return text[open_brace + 1:].strip()


print("=" * 70)
print("CODE-EVOL-200 + SELF-SCoT — MBCPP")
print("=" * 70)

print("\nLoading MBCPP...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    problems = [json.loads(line) for line in f if line.strip()]

print(f"Problems: {len(problems)}")
print(f"Candidates/problem: {NUM_SAMPLES}")
print("Metric: pass@1")

assert len(problems) == 848, (
    f"Expected 848 MBCPP problems, found {len(problems)}"
)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("Loading CodeEvol-200 LoRA...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("Model ready.\n")

results = []

empty_scot_count = 0
empty_code_count = 0
scot_fence_count = 0
code_fence_count = 0
full_function_count = 0


for i, problem in enumerate(problems):

    task_id = problem["task_id"]
    entry_point = problem["entry_point"]
    prompt = problem["prompt"]

    print(f"[{i + 1}/{len(problems)}] {task_id}")

    # ==============================================================
    # STAGE 1: GENERATE SELF-SCoT
    # ==============================================================

    scot_messages = [
        {
            "role": "user",
            "content": (
                "Analyze the following C++ programming problem. "
                "Think through the algorithm, important edge cases, "
                "and the implementation steps needed to solve it. "
                "Do not write the final C++ code. "
                "Provide concise step-by-step reasoning.\n\n"
                + prompt
            )
        }
    ]

    scot_prompt = tokenizer.apply_chat_template(
        scot_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    scot_inputs = tokenizer(
        scot_prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        scot_output = model.generate(
            **scot_inputs,
            max_new_tokens=MAX_SCOT_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    scot_generated = scot_output[
        0
    ][scot_inputs["input_ids"].shape[1]:]

    scot = tokenizer.decode(
        scot_generated,
        skip_special_tokens=True
    ).strip()

    if "```" in scot:
        scot_fence_count += 1

    if not scot:
        empty_scot_count += 1

    del scot_inputs, scot_output, scot_generated
    torch.cuda.empty_cache()

    print("  SCoT generated.")

    # ==============================================================
    # STAGE 2: GENERATE CODE USING THE GENERATED SCoT
    # ==============================================================

    code_messages = [
        {
            "role": "user",
            "content": (
                "Complete the following C++ function. "
                "The function declaration is already provided. "
                "Return only the function body. "
                "Do not repeat the function declaration. "
                "Do not use markdown code fences.\n\n"
                "PROBLEM:\n"
                + prompt
                + "\n\n"
                "REASONING:\n"
                + scot
                + "\n\n"
                "Now write the C++ function body based on "
                "the problem and the reasoning above."
            )
        }
    ]

    code_prompt = tokenizer.apply_chat_template(
        code_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    code_inputs = tokenizer(
        code_prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        code_output = model.generate(
            **code_inputs,
            max_new_tokens=MAX_CODE_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            num_return_sequences=NUM_SAMPLES,
            pad_token_id=tokenizer.eos_token_id
        )

    code_generated = code_output[
        0
    ][code_inputs["input_ids"].shape[1]:]

    raw_completion = tokenizer.decode(
        code_generated,
        skip_special_tokens=True
    ).strip()

    if "```" in raw_completion:
        code_fence_count += 1

    if re.search(
        rf"\b{re.escape(entry_point)}\s*\([^{{}}]*\)\s*\{{",
        raw_completion,
        flags=re.S
    ):
        full_function_count += 1

    completion = clean_completion(
        raw_completion,
        entry_point
    )

    if not completion:
        empty_code_count += 1

    results.append({
        "task_id": task_id,
        "language": problem["language"],
        "scot": scot,
        "completion": completion
    })

    del code_inputs, code_output, code_generated
    torch.cuda.empty_cache()

    print("  Code generated.")


# ==============================================================
# SAVE
# ==============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")


print("\n" + "=" * 70)
print("CODE-EVOL-200 + SELF-SCoT MBCPP GENERATION COMPLETE")
print("=" * 70)
print(f"Problems: {len(results)}")
print(f"Candidates/problem: {NUM_SAMPLES}")
print("Metric: pass@1")
print(f"SCoT markdown cases: {scot_fence_count}")
print(f"Code markdown cases: {code_fence_count}")
print(f"Full functions detected: {full_function_count}")
print(f"Empty SCoTs: {empty_scot_count}")
print(f"Empty completions: {empty_code_count}")
print(f"Saved: {OUTPUT_FILE}")
