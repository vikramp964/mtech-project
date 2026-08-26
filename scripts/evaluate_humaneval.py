import json
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

LORA_PATH = "/workspace/project/outputs/qwen-codealpaca-lora-full"

RESULTS_DIR = "/workspace/project/results"

USE_LORA = False

MAX_NEW_TOKENS = 256


# --------------------------------------------------
# Load HumanEval
# --------------------------------------------------

print("Loading HumanEval dataset...")

dataset = load_dataset("openai/openai_humaneval")

problems = dataset["test"]

print(f"Total problems: {len(problems)}")



# --------------------------------------------------
# Load Tokenizer
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --------------------------------------------------
# Load Base Model
# --------------------------------------------------

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

model.eval()


# --------------------------------------------------
# Load LoRA Adapter (Optional)
# --------------------------------------------------

if USE_LORA:

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        LORA_PATH,
    )

    print("LoRA adapter loaded successfully!")

else:

    print("Using base model.")


# --------------------------------------------------
# Generate Solution
# --------------------------------------------------

def generate_solution(prompt):

    inputs = tokenizer(
        prompt,
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

    generated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return generated


# --------------------------------------------------
# Evaluate HumanEval
# --------------------------------------------------

print("\nStarting HumanEval evaluation...\n")

predictions = []

for i, problem in enumerate(problems):

    print(f"[{i+1}/{len(problems)}] {problem['task_id']}")

    prompt = problem["prompt"]

    generated_code = generate_solution(prompt)

    predictions.append(
        {
            "task_id": problem["task_id"],
            "prompt": prompt,
            "completion": generated_code,
        }
    )

print("\nEvaluation completed.")


# --------------------------------------------------
# Save Predictions
# --------------------------------------------------

import os

os.makedirs(RESULTS_DIR, exist_ok=True)

if USE_LORA:
    output_file = os.path.join(
        RESULTS_DIR,
        "lora_predictions.jsonl",
    )
else:
    output_file = os.path.join(
        RESULTS_DIR,
        "base_predictions.jsonl",
    )

print(f"\nSaving predictions to:\n{output_file}")

with open(output_file, "w") as f:

    for prediction in predictions:

        f.write(json.dumps(prediction) + "\n")

print("\nPredictions saved successfully!")
