import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = "/workspace/project/models/Qwen2.5-Coder-32B-Instruct"

INPUT_FILE = "/workspace/project/datasets/code_alpaca_500_train.json"
OUTPUT_FILE = "/workspace/project/results/scot_teacher_codealpaca_450.jsonl"

NUM_TEST = 450
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = """You are an expert programming teacher.

Given a programming instruction and optional input, produce a concise
Structured Chain of Thought (SCoT) that explains the solution strategy.

The reasoning must be useful for another programming model that will
later generate the answer.

Do NOT write the final answer or code.
Do NOT assume a particular programming language unless the task specifies it.

Use this format:

Input:
...

Output:
...

1. ...
2. ...
3. ...

Keep the reasoning concise, structured, and directly relevant to the task.
"""

print("=" * 70)
print("CODE-ALPACA SCoT TEACHER TEST")
print("=" * 70)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

with open(INPUT_FILE, encoding="utf-8") as f:
    data = json.load(f)

print(f"Training examples available: {len(data)}")
print(f"Testing first {NUM_TEST} examples.")

# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

# ------------------------------------------------------------
# Load teacher
# ------------------------------------------------------------

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
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

model.eval()

print("Teacher loaded successfully.")

# ------------------------------------------------------------
# Generate SCoT
# ------------------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for i, example in enumerate(data[:NUM_TEST]):

        print(f"\n[{i + 1}/{NUM_TEST}] Generating SCoT")

        instruction = example["instruction"]
        input_text = example["input"]

        user_content = f"""Programming Instruction:

{instruction}
"""

        if input_text.strip():
            user_content += f"""
Input:

{input_text}
"""

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
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
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        scot = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        record = {
            "instruction": instruction,
            "input": input_text,
            "scot": scot,
            "output": example["output"],
        }

        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

        print("SCoT generated.")

print("\n" + "=" * 70)
print("SCoT TEST COMPLETE")
print("=" * 70)
print(f"Saved: {OUTPUT_FILE}")
