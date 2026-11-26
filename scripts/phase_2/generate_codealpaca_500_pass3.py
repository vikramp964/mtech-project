import json
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
LORA_PATH = "/workspace/project/outputs/qwen-codealpaca-500-lora"

OUTPUT_FILE = "/workspace/project/results/codealpaca_500_pass3_predictions.jsonl"

MAX_NEW_TOKENS = 256
NUM_SAMPLES = 3
TEMPERATURE = 0.7

print("=" * 70)
print("CODE-ALPACA-500 PASS@3 GENERATION")
print("=" * 70)

print("\nLoading HumanEval...")
dataset = load_dataset("openai/openai_humaneval")
problems = dataset["test"]

print(f"Problems: {len(problems)}")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nLoading Qwen-7B...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

print("Loading Code-Alpaca-500 LoRA...")
model = PeftModel.from_pretrained(
    model,
    LORA_PATH,
)

model.eval()

print("\nGenerating 3 candidates per problem...\n")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for i, problem in enumerate(problems):

        print(f"[{i+1}/{len(problems)}] {problem['task_id']}")

        inputs = tokenizer(
            problem["prompt"],
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                num_return_sequences=NUM_SAMPLES,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[-1]

        completions = []

        for output in outputs:
            completions.append(
                tokenizer.decode(
                    output[input_length:],
                    skip_special_tokens=True,
                )
            )

        f.write(json.dumps({
            "task_id": problem["task_id"],
            "prompt": problem["prompt"],
            "completions": completions,
        }) + "\n")

print("\n" + "=" * 70)
print("CODE-ALPACA-500 PASS@3 GENERATION COMPLETE")
print("=" * 70)
print(f"Saved: {OUTPUT_FILE}")
