import json
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "/workspace/project/models/Qwen2.5-Coder-32B-Instruct"

OUTPUT_FILE = "/workspace/project/results/scot_teacher_humaneval.jsonl"

NUM_TEST_PROBLEMS = 164

# ============================================================
# Pure SCoT few-shot prompt
# ============================================================

SCOT_PROMPT = r"""
You are an expert programmer generating Structured Chain-of-Thought
(SCoT) for Python programming problems.

Your task is to produce a structured solution plan for the given
programming requirement before code generation.

The SCoT should:

1. Identify the input and output.
2. Describe the solution using program structures:
   - Sequence
   - Branch
   - Loop
3. Show nesting when one structure occurs inside another.
4. Describe the required operations clearly.
5. Do NOT write actual Python code.
6. Do NOT use programming syntax as pseudocode.
7. Return only the SCoT.

Use the following examples as demonstrations.

============================================================
EXAMPLE 1 — SEQUENCE
============================================================

Requirement:
Write a Python function that calculates the average of all
numbers in a list.

SCoT:

Input: nums: list[float]
Output: result: float

1. Calculate the sum of all numbers in nums.
2. Calculate the number of elements in nums.
3. Divide the sum by the number of elements.
4. Return the resulting average.


============================================================
EXAMPLE 2 — BRANCH
============================================================

Requirement:
Write a Python function that returns the absolute value
of a number.

SCoT:

Input: n: int or float
Output: result: int or float

1. If n is greater than or equal to 0:
   2. Return n.
3. Else:
   4. Return the negation of n.


============================================================
EXAMPLE 3 — LOOP
============================================================

Requirement:
Write a Python function that counts the number of even
elements in a list.

SCoT:

Input: nums: list[int]
Output: count: int

1. Initialize count to 0.
2. For each number in nums:
   3. Determine whether the number is even.
   4. If the number is even:
      5. Increment count by 1.
6. Return count.


============================================================
EXAMPLE 4 — BRANCH + LOOP
============================================================

Requirement:
Write a Python function that returns the first repeated
element in a list. If there is no repeated element,
return None.

SCoT:

Input: nums: list[int]
Output: result: int or None

1. Initialize an empty set called seen.
2. For each number in nums:
   3. If the number is already present in seen:
      4. Return the number.
   5. Else:
      6. Add the number to seen.
7. Return None.


============================================================
NEW REQUIREMENT
============================================================

{PROBLEM}

============================================================
GENERATE THE SCoT
============================================================
"""


# ============================================================
# Load teacher
# ============================================================

print("=" * 70)
print("SCoT-ONLY TEACHER GENERATION")
print("=" * 70)

print("\nLoading HumanEval...")

dataset = load_dataset("openai/openai_humaneval")
problems = dataset["test"]

print(f"Total HumanEval problems: {len(problems)}")
print(f"Testing first {NUM_TEST_PROBLEMS} problems only.")

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

print("\nLoading Qwen2.5-Coder-32B teacher...")

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
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

model.eval()

print("\nTeacher loaded successfully.")


# ============================================================
# Generate SCoT
# ============================================================

print("\nGenerating SCoT examples...\n")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for i in range(NUM_TEST_PROBLEMS):
        problem = problems[i]

        print(f"[{i + 1}/{NUM_TEST_PROBLEMS}] {problem['task_id']}")

        prompt = SCOT_PROMPT.format(
            PROBLEM=problem["prompt"]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert programmer and "
                    "programming instructor."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
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
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        record = {
            "task_id": problem["task_id"],
            "prompt": problem["prompt"],
            "scot": generated,
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("SCoT generated.")


print("\n" + "=" * 70)
print("SCoT TEST GENERATION COMPLETE")
print("=" * 70)
print(f"Saved: {OUTPUT_FILE}")
