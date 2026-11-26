import json
import os
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

MAX_NEW_TOKENS = 256
NUM_SAMPLES = 3
TEMPERATURE = 0.7

RESULTS_DIR = "/workspace/project/results"

MODELS = {
    "base": None,
    "alpaca": "/workspace/project/outputs/qwen-codealpaca-lora-full",
    "evol_scot": "/workspace/project/outputs/qwen-codeevol-scot-lora",
}


print("Loading HumanEval...")
dataset = load_dataset("openai/openai_humaneval")
problems = dataset["test"]

print(f"Total problems: {len(problems)}")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def load_model(adapter_path):

    print("\nLoading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    if adapter_path is not None:
        print(f"Loading LoRA: {adapter_path}")

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
        )

    model.eval()

    return model


def generate_candidates(model, prompt):

    inputs = tokenizer(
        prompt,
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

    candidates = []

    for output in outputs:

        completion = tokenizer.decode(
            output[input_length:],
            skip_special_tokens=True,
        )

        candidates.append(completion)

    return candidates


os.makedirs(RESULTS_DIR, exist_ok=True)


for model_name, adapter_path in MODELS.items():

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    model = load_model(adapter_path)

    output_file = os.path.join(
        RESULTS_DIR,
        f"{model_name}_pass2_predictions.jsonl",
    )

    with open(output_file, "w", encoding="utf-8") as f:

        for i, problem in enumerate(problems):

            print(
                f"[{i + 1}/{len(problems)}] "
                f"{problem['task_id']}"
            )

            candidates = generate_candidates(
                model,
                problem["prompt"],
            )

            record = {
                "task_id": problem["task_id"],
                "prompt": problem["prompt"],
                "completions": candidates,
            }

            f.write(
                json.dumps(record) + "\n"
            )

    print(f"\nSaved: {output_file}")

    del model
    torch.cuda.empty_cache()

print("\nPASS@2 GENERATION COMPLETE")
