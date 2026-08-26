import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_PATH = "/workspace/project/models/Qwen2.5-Coder-32B-Instruct"

print("=" * 60)
print("Loading tokenizer...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

print("=" * 60)
print("Loading model...")
print("=" * 60)

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

print("\nTeacher loaded successfully!\n")

messages = [
    {
        "role": "system",
        "content": (
            "You are an expert competitive programmer "
            "and programming instructor."
        ),
    },
    {
        "role": "user",
        "content": (
            "Write a C++ function to reverse a linked list."
        ),
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

print("=" * 60)
print("Generating...")
print("=" * 60)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0,
    )

answer = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
)

print("\n")
print(answer)
