import json
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

OUTPUT_FILE = "/workspace/project/results/codeevol_200_scot_humaneval_predictions.jsonl"

MAX_SCOT_TOKENS = 256
MAX_CODE_TOKENS = 256

def clean_code(text):
    text = text.strip()

    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


print("=" * 70)
print("CODE-EVOL-200 + SELF-SCoT HUMAN EVAL")
print("=" * 70)

# ============================================================
# Load HumanEval
# ============================================================

print("\nLoading HumanEval...")

dataset = load_dataset("openai/openai_humaneval")
NUM_TEST_PROBLEMS = 164
problems = dataset["test"].select(range(NUM_TEST_PROBLEMS))

print(f"Problems: {len(problems)}")

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
# Load SAME 7B model
# ============================================================

print("\nLoading Qwen2.5-Coder-7B-Instruct...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

model = PeftModel.from_pretrained(
    model,
    "/workspace/project/outputs/qwen-codeevol-200-lora",
)

model.eval()

print("Code-Evol-200 LoRA loaded successfully.")

# ============================================================
# SCoT generation prompt
# ============================================================

SCOT_SYSTEM = """You are an expert competitive programmer.

Analyze the programming problem using Structured Chain of Thought (SCoT).

Represent the solution using:
- Sequence: the ordered operations
- Branch: important conditions or decisions
- Loop: repetitions or iterations

Do NOT write the final code.
Produce only a concise structured solution plan."""

# ============================================================
# Code generation prompt
# ============================================================

CODE_SYSTEM = """You are an expert competitive programmer.

Write the correct Python solution for the given programming problem.

Use the provided Structured Chain of Thought as guidance.

Return ONLY the Python code.
Do not explain the solution.
Do not use markdown code fences."""

# ============================================================
# Generate
# ============================================================

print("\nGenerating Code-Evol-200 + Self-SCoT solutions...\n")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for i, problem in enumerate(problems):

        print(f"[{i + 1}/{len(problems)}] {problem['task_id']}")

        original_prompt = problem["prompt"]

        # ----------------------------------------------------
        # Stage 1: SAME 7B generates SCoT
        # ----------------------------------------------------

        scot_messages = [
            {
                "role": "system",
                "content": SCOT_SYSTEM,
            },
            {
                "role": "user",
                "content": (
                    "Programming Problem:\n\n"
                    + original_prompt
                    + "\n\n"
                    "Generate the Structured Chain of Thought."
                ),
            },
        ]

        scot_text = tokenizer.apply_chat_template(
            scot_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        scot_inputs = tokenizer(
            scot_text,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():

            scot_outputs = model.generate(
                **scot_inputs,
                max_new_tokens=MAX_SCOT_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        scot_input_length = scot_inputs["input_ids"].shape[-1]

        scot = tokenizer.decode(
            scot_outputs[0][scot_input_length:],
            skip_special_tokens=True,
        ).strip()

        print("  SCoT generated.")

        # ----------------------------------------------------
        # Stage 2: SAME 7B generates code using SCoT
        # ----------------------------------------------------

        code_messages = [
            {
                "role": "system",
                "content": CODE_SYSTEM,
            },
            {
                "role": "user",
                "content": (
                    "Programming Problem:\n\n"
                    + original_prompt
                    + "\n\n"
                    "Structured Chain of Thought:\n\n"
                    + scot
                    + "\n\n"
                    "Now generate the final Python solution."
                ),
            },
        ]

        code_text = tokenizer.apply_chat_template(
            code_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        code_inputs = tokenizer(
            code_text,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():

            code_outputs = model.generate(
                **code_inputs,
                max_new_tokens=MAX_CODE_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        code_input_length = code_inputs["input_ids"].shape[-1]

        completion = tokenizer.decode(
            code_outputs[0][code_input_length:],
            skip_special_tokens=True,
        )

        completion = clean_code(completion)

        print("  Code generated.")

        record = {
            "task_id": problem["task_id"],
            "prompt": original_prompt,
            "scot": scot,
            "completion": completion,
        }

        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

print("\n" + "=" * 70)
print("CODE-EVOL-200 + SELF-SCoT GENERATION COMPLETE")
print("=" * 70)
print(f"Saved: {OUTPUT_FILE}")
