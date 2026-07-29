import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "/workspace/project/outputs/qwen-codealpaca-lora-test"

prompts = [
    "Write a Python function that returns the second largest distinct number in a list.",
    "Write a Python function to check whether two strings are anagrams.",
    "Write a Python function that finds the longest common prefix of a list of strings."
]

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)


def generate(model, prompt):

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    )


# ----------------------------------------
# BASE MODEL
# ----------------------------------------

print("\n========== BASE MODEL ==========\n")

base_outputs = []

for prompt in prompts:

    response = generate(base_model, prompt)
    base_outputs.append(response)

    print("\nPROMPT:")
    print(prompt)

    print("\nBASE RESPONSE:")
    print(response)

    print("\n" + "=" * 70)


# ----------------------------------------
# LOAD LoRA
# ----------------------------------------

print("\nLoading LoRA adapter...")

lora_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

lora_model.eval()


# ----------------------------------------
# LoRA MODEL
# ----------------------------------------

print("\n========== LoRA MODEL ==========\n")

for prompt, base_response in zip(prompts, base_outputs):

    lora_response = generate(lora_model, prompt)

    print("\nPROMPT:")
    print(prompt)

    print("\nBASE:")
    print(base_response)

    print("\nLoRA:")
    print(lora_response)

    print("\n" + "=" * 70)
