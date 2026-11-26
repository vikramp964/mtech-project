import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "/workspace/project/outputs/qwen-codeevol-200-lora"
INPUT_FILE = "/workspace/project/external/mxeval/data/mbxp/mbcpp_release_v1.2.jsonl"

MAX_SCOT_TOKENS = 256
MAX_CODE_TOKENS = 256
TEMPERATURE = 0.7

# Use exactly one task.
TASK_INDEX = 0


def generate(model, tokenizer, messages, max_new_tokens):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = output[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    del inputs, output, generated
    torch.cuda.empty_cache()

    return prompt, text


print("=" * 70)
print("SINGLE-TASK SELF-SCoT VERIFICATION — MBCPP")
print("=" * 70)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    problems = [json.loads(line) for line in f if line.strip()]

problem = problems[TASK_INDEX]

task_id = problem["task_id"]
prompt = problem["prompt"]

print(f"\nTask: {task_id}")
print(f"Language: {problem['language']}")
print(f"Entry point: {problem['entry_point']}")

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

print("Model ready.")


# ==============================================================
# STAGE 1 — GENERATE SCoT
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

scot_prompt, scot = generate(
    model,
    tokenizer,
    scot_messages,
    MAX_SCOT_TOKENS
)

print("\n" + "=" * 70)
print("STAGE 1 — GENERATED SCoT")
print("=" * 70)
print(scot)

if not scot:
    raise RuntimeError("SCoT is EMPTY. Stop — do not run full experiment.")

print("\nSCoT status: NON-EMPTY")


# ==============================================================
# STAGE 2 — GENERATE CODE USING SCoT
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

code_prompt, completion = generate(
    model,
    tokenizer,
    code_messages,
    MAX_CODE_TOKENS
)

print("\n" + "=" * 70)
print("STAGE 2 — ACTUAL CODE-GENERATION PROMPT")
print("=" * 70)
print(code_prompt)

print("\n" + "=" * 70)
print("STAGE 2 — GENERATED CODE")
print("=" * 70)
print(completion)

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

checks = {
    "SCoT generated": bool(scot.strip()),
    "Problem included in Stage 2": prompt in code_prompt,
    "SCoT included in Stage 2": scot in code_prompt,
    "Code generated": bool(completion.strip()),
}

for name, result in checks.items():
    print(f"{name}: {'PASS' if result else 'FAIL'}")

if not all(checks.values()):
    raise RuntimeError(
        "\nSCoT pipeline verification FAILED. "
        "Do not run the 848-task experiment."
    )

print("\nALL CHECKS PASSED.")
print("The generated SCoT is actually being supplied to Stage 2.")
print("Safe to proceed to the full 848-task experiment.")
