import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

TRAIN_DATA_PATH = "/workspace/project/datasets/code_alpaca_500_scot_train.json"
VALID_DATA_PATH = "/workspace/project/datasets/code_alpaca_500_valid.json"

OUTPUT_DIR = "/workspace/project/outputs/qwen-codealpaca-500-scot-lora"
MAX_LENGTH = 512


# --------------------------------------------------
# 1. Load Code Alpaca
# --------------------------------------------------

print("Loading training dataset...")

with open(TRAIN_DATA_PATH, "r") as f:
    train_data = json.load(f)

print("Loading validation dataset...")

with open(VALID_DATA_PATH, "r") as f:
    valid_data = json.load(f)

print(f"Training samples   : {len(train_data)}")
print(f"Validation samples : {len(valid_data)}")



# --------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token




# --------------------------------------------------
# 3. Format Code Alpaca examples
# --------------------------------------------------

def format_example(example):

    instruction = example["instruction"]
    input_text = example["input"]
    scot = example["scot"]
    output = example["output"]

    if input_text.strip():
        user_content = (
            f"{instruction}\n\n"
            f"Input:\n{input_text}\n\n"
            f"Structured Chain of Thought:\n{scot}"
        )
    else:
        user_content = (
            f"{instruction}\n\n"
            f"Structured Chain of Thought:\n{scot}"
        )

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


train_texts = [format_example(x) for x in train_data]

# Keep validation identical to the original Code-Alpaca-500 experiment.
# Validation examples do not contain teacher SCoT.
def format_valid_example(example):

    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    if input_text.strip():
        user_content = (
            f"{instruction}\n\n"
            f"Input:\n{input_text}"
        )
    else:
        user_content = instruction

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

valid_texts = [format_valid_example(x) for x in valid_data]

train_dataset = Dataset.from_dict({"text": train_texts})
valid_dataset = Dataset.from_dict({"text": valid_texts})

# --------------------------------------------------
# 4. Tokenize
# --------------------------------------------------

def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


tokenized_train = train_dataset.map(
    tokenize,
    remove_columns=["text"],
)

tokenized_valid = valid_dataset.map(
    tokenize,
    remove_columns=["text"],
)


# --------------------------------------------------
# 5. Load Qwen
# --------------------------------------------------

print("Loading Qwen model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
model.config.use_cache = False
model.enable_input_require_grads()


# --------------------------------------------------
# 6. Configure LoRA
# --------------------------------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# --------------------------------------------------
# 7. Training configuration
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=1,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    bf16=True,
    fp16=False,

    logging_strategy="steps",
    logging_steps=10,

    save_strategy="epoch",
    eval_strategy="epoch",
    eval_on_start=False,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    save_total_limit=2,

    report_to="none",

    remove_unused_columns=False,

)


# --------------------------------------------------
# 8. Data collator
# --------------------------------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# --------------------------------------------------
# 9. Trainer
# --------------------------------------------------



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
)

# --------------------------------------------------
# 10. Train
# --------------------------------------------------

print("\nStarting LoRA training...\n")

trainer.train()
trainer.save_state()
print("\nRunning validation...\n")

metrics = trainer.evaluate()

print(metrics)

# --------------------------------------------------
# 11. Save LoRA adapter
# --------------------------------------------------

print("\nSaving LoRA adapter...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining + Validation complete.")
print(f"Adapter saved to: {OUTPUT_DIR}")
