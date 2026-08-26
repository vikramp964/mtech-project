import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "/workspace/project/outputs/qwen-codeevol-200-lora"

INPUT_FILE = "/workspace/project/external/mxeval/data/mbxp/mbcpp_release_v1.2.jsonl"
OUTPUT_FILE = "/workspace/project/results/codeevol_200_mbcpp_predictions.jsonl"

MAX_CODE_TOKENS = 256
TEMPERATURE = 0.7
NUM_SAMPLES = 1


def clean_completion(text, entry_point):
    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(r"^\s*```(?:cpp|c\+\+)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)

    text = text.strip()

    # If the model generated the complete function, extract its body.
    pattern = rf"\b{re.escape(entry_point)}\s*\([^{{}}]*\)\s*\{{"
    match = re.search(pattern, text, flags=re.S)

    if match:
        open_brace = text.find("{", match.start())

        depth = 0

        for i in range(open_brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1

                if depth == 0:
                    return text[open_brace + 1:i].strip()

    return text


print("=" * 70)
print("CODE-EVOL-200 — MBCPP BASELINE")
print("=" * 70)

print("\nLoading MBCPP...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    problems = [json.loads(line) for line in f if line.strip()]

print(f"Problems: {len(problems)}")
print(f"Candidates/problem: {NUM_SAMPLES}")
print("Metric: pass@1")

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

empty_count = 0
fence_count = 0
full_function_count = 0

for i, problem in enumerate(problems):

    task_id = problem["task_id"]
    entry_point = problem["entry_point"]

    print(f"[{i + 1}/{len(problems)}] {task_id}")

    prompt = problem["prompt"]

    messages = [
        {
            "role": "user",
            "content": (
                "Complete the following C++ function. "
                "The function declaration is already provided. "
                "Return only the function body. "
                "Do not repeat the function declaration. "
                "Do not use markdown code fences.\n\n"
                + prompt
            )
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_CODE_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            num_return_sequences=NUM_SAMPLES,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = output[0][inputs["input_ids"].shape[1]:]

    raw_completion = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    if "```" in raw_completion:
        fence_count += 1

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
        empty_count += 1

    results.append({
        "task_id": task_id,
        "language": problem["language"],
        "completion": completion
    })

    del inputs, output, generated
    torch.cuda.empty_cache()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")

print("\n" + "=" * 70)
print("MBCPP BASELINE GENERATION COMPLETE")
print("=" * 70)
print(f"Problems: {len(results)}")
print(f"Candidates/problem: {NUM_SAMPLES}")
print("Metric: pass@1")
print(f"Markdown cases cleaned: {fence_count}")
print(f"Full functions detected: {full_function_count}")
print(f"Empty completions: {empty_count}")
print(f"Saved: {OUTPUT_FILE}")
