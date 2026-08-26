import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
LORA_PATH = "/workspace/project/outputs/qwen-codealpaca-lora-full"

# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    LORA_PATH,
    local_files_only=True,
)

# --------------------------------------------------
# Load base model
# --------------------------------------------------

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

# --------------------------------------------------
# Load LoRA adapter
# --------------------------------------------------

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    LORA_PATH,
)

model.eval()

print("\nModel loaded successfully!\n")

# --------------------------------------------------
# Interactive loop
# --------------------------------------------------

while True:

    prompt = input("\nYou: ")

    if prompt.lower() in ["exit", "quit"]:
        break

    messages = [
        {"role": "user", "content": prompt},
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
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
        )

    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    print("\nAssistant:\n")
    print(answer)
