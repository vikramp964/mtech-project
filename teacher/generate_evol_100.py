import json
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_PATH = "/workspace/project/models/Qwen2.5-Coder-32B-Instruct"

INPUT_FILE = "/workspace/project/datasets/code_alpaca_500_train.json"
OUTPUT_FILE = "/workspace/project/datasets/code_alpaca_100_evolved.json"

NUM_SAMPLES = 100
MAX_NEW_TOKENS = 900

SYSTEM_PROMPT = """You are an expert competitive programming problem designer.

Your task is to evolve an existing programming problem using the
Code Evol-Instruct method used in WizardCoder.

Use EXACTLY this evolution strategy:

Add new constraints and requirements to the original problem,
adding approximately 10 additional words.

The evolved problem must:
1. remain directly related to the original problem,
2. be slightly harder than the original,
3. add meaningful constraints or requirements,
4. remain solvable,
5. preserve the original programming language when applicable.

After creating the evolved problem, provide a correct solution.

Return ONLY:

Enhanced Instruction:
...

Solution:
...
"""

def clean_code(text):
    text = text.strip()

    if "```" in text:
        text = re.sub(r"```[a-zA-Z0-9_+-]*", "", text)
        text = text.replace("```", "")

    return text.strip()


def build_prompt(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")

    if input_text.strip():
        original = (
            "Instruction:\n"
            + instruction
            + "\n\nInput:\n"
            + input_text
        )
    else:
        original = instruction

    return (
        "Original Programming Problem:\n\n"
        + original
        + "\n\n"
        "Evolve this problem using ONLY the specified evolution strategy."
    )


def parse_response(response):
    if "Enhanced Instruction:" not in response:
        return "", ""

    section = response.split("Enhanced Instruction:", 1)[1]

    if "Solution:" in section:
        enhanced, solution = section.split("Solution:", 1)
    else:
        enhanced = section
        solution = ""

    return enhanced.strip(), clean_code(solution)


print("=" * 70)
print("WIZARDCODER CODE EVOL-INSTRUCT — 100 SAMPLE PILOT")
print("=" * 70)

print("\nLoading Code-Alpaca...")

with open(INPUT_FILE, encoding="utf-8") as f:
    dataset = json.load(f)

examples = dataset[:NUM_SAMPLES]

print(f"Source dataset: {len(dataset)}")
print(f"Using first: {len(examples)} samples")

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nLoading Qwen2.5-Coder-32B teacher in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)

model.eval()

print("Teacher loaded successfully.")

results = []
failures = []

print("\nStarting evolution...\n")

for i, example in enumerate(examples):

    print(f"[{i + 1}/{NUM_SAMPLES}] Generating...")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_prompt(example),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    ).strip()

    enhanced, solution = parse_response(response)

    if not enhanced or not solution:
        print("  WARNING: failed to parse.")
        failures.append({
            "index": i,
            "response": response,
        })
        continue

    record = {
        "source_index": i,
        "original_instruction": example.get("instruction", ""),
        "original_input": example.get("input", ""),
        "original_output": example.get("output", ""),
        "evolution_method": (
            "Add new constraints and requirements "
            "to the original problem"
        ),
        "enhanced_instruction": enhanced,
        "evolved_output": solution,
    }

    results.append(record)

    print("  Evolved successfully.")

print("\nWriting results...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\n" + "=" * 70)
print("EVOLUTION COMPLETE")
print("=" * 70)
print(f"Successful: {len(results)}")
print(f"Failed:     {len(failures)}")
print(f"Saved:      {OUTPUT_FILE}")

if failures:
    failure_file = OUTPUT_FILE.replace(
        ".json",
        "_failures.json",
    )

    with open(failure_file, "w", encoding="utf-8") as f:
        json.dump(
            failures,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Failures:   {failure_file}")
