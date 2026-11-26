import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

SCOT_FILE = "/workspace/project/results/scot_teacher_humaneval.jsonl"
OUTPUT_FILE = "/workspace/project/results/base_scot_predictions.jsonl"

MAX_NEW_TOKENS = 256


def clean_completion(text):
    text = text.strip()

    # Remove markdown code fences without writing backtick literals
    if text.startswith(chr(96) * 3 + "python"):
        text = text[10:].strip()
    elif text.startswith(chr(96) * 3):
        text = text[3:].strip()

    if text.endswith(chr(96) * 3):
        text = text[:-3].strip()

    return text


print("=" * 70)
print("BASE QWEN-7B + SCoT GENERATION")
print("=" * 70)

# ============================================================
# Load teacher-generated SCoTs
# ============================================================

print("\nLoading teacher SCoTs...")

with open(SCOT_FILE, encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

print(f"Problems: {len(data)}")

# ============================================================
# Load tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# Load BASE model
# ============================================================

print("\nLoading BASE Qwen2.5-Coder-7B...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

model.eval()

print("\nBase model loaded successfully.")

# ============================================================
# Student prompt
# ============================================================

def build_prompt(problem, scot):

    return f"""You are an expert Python programmer.

Generate a correct Python solution for the programming
requirement below.

A Structured Chain-of-Thought (SCoT) has been provided as
a solution plan. Follow the SCoT carefully when generating
the solution.

Return ONLY the Python code.

============================================================
REQUIREMENT
============================================================

{problem}

============================================================
STRUCTURED CHAIN-OF-THOUGHT
============================================================

{scot}

============================================================
GENERATE CODE
============================================================
"""

# ============================================================
# Generate solutions
# ============================================================

print("\nGenerating solutions...\n")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for i, item in enumerate(data):

        print(f"[{i + 1}/{len(data)}] {item['task_id']}")

        prompt = build_prompt(
            item["prompt"],
            item["scot"],
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[-1]

        completion = tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )

        completion = clean_completion(completion)

        record = {
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "scot": item["scot"],
            "completion": completion,
        }

        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

print("\n" + "=" * 70)
print("BASE + SCoT GENERATION COMPLETE")
print("=" * 70)
print(f"Saved: {OUTPUT_FILE}")
