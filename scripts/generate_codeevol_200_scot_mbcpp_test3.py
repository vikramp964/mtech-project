import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "/workspace/project/outputs/qwen-codeevol-200-lora"

INPUT_FILE = "/workspace/project/external/mxeval/data/mbxp/mbcpp_release_v1.2.jsonl"
OUTPUT_FILE = "/workspace/project/results/codeevol_200_scot_mbcpp_predictions.jsonl"

# ==============================================================
# ORIGINAL SCoT PAPER SETTINGS
# ==============================================================

MAX_SCOT_TOKENS = 300
MAX_CODE_TOKENS = 300

SCOT_TEMPERATURE = 0.8
SCOT_TOP_P = 0.95

CODE_TEMPERATURE = 0.0

NUM_SCOTS = 1

# ==============================================================
# SCoT FEW-SHOT EXAMPLES
#
# The original paper uses 3 examples by default.
# These examples demonstrate Input/Output + sequence/branch/loop
# structures. They are based on the examples shown in the paper.
# ==============================================================

SCOT_EXAMPLES = [
    {
        "problem": """Write a function to find the first repeated character in a string.""",

        "scot": """Input: str: string
Output: ch: the first repeated character or None

1: for each character ch in str:
2:     if ch appears more than once in str:
3:         return ch
4: return None""",

        "code": """def first_Repeated_Char(str):
    h = {}
    for ch in str:
        if ch in h:
            return ch
        else:
            h[ch] = 0
    return None"""
    },

    {
        "problem": """Write a function to find sequences of lowercase letters joined with an underscore.""",

        "scot": """Input: text: string
Output: seq: list of strings

1: Initialize seq as an empty list
2: for each word in text split by space:
3:     if word matches the pattern of lowercase letters joined with an underscore:
4:         append word to seq
5: return seq""",

        "code": """def text_lowercase_underscore(text):
    seq = []
    for word in text.split():
        if re.match(r'^[a-z]+_[a-z]+$', word):
            seq.append(word)
    return seq"""
    },

    {
        "problem": """Write a function to determine whether a list is monotonically increasing or decreasing.""",

        "scot": """Input: l: list
Output: True or False

1: Initialize increasing to False
2: Initialize decreasing to False
3: for each element after the first element in l:
4:     if the current element is greater than the previous element:
5:         set increasing to True
6:     if the current element is less than the previous element:
7:         set decreasing to True
8:     if both increasing and decreasing are True:
9:         return False
10: return True""",

        "code": """def monotonic(l):
    increasing = False
    decreasing = False

    for i in range(1, len(l)):
        if l[i] > l[i-1]:
            increasing = True
        if l[i] < l[i-1]:
            decreasing = True
        if increasing and decreasing:
            return False

    return True"""
    }
]


def clean_completion(text, entry_point):
    text = text.strip()

    text = re.sub(
        r"^\s*```(?:cpp|c\+\+)?\s*",
        "",
        text,
        flags=re.I
    )

    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    pattern = (
        rf"\b{re.escape(entry_point)}\s*"
        r"\([^{}]*\)\s*\{"
    )

    match = re.search(pattern, text, flags=re.S)

    if not match:
        return text

    open_brace = text.find("{", match.start())
    depth = 0

    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1

            if depth == 0:
                return text[open_brace + 1:i].strip()

    return text[open_brace + 1:].strip()


def build_scot_prompt(problem_prompt):
    examples = ""

    for ex in SCOT_EXAMPLES:
        examples += (
            "\n\nEXAMPLE REQUIREMENT:\n"
            + ex["problem"]
            + "\n\n"
            "EXAMPLE SCoT:\n"
            + ex["scot"]
        )

    return (
        "Please understand the requirement and write a rough solving process. "
        "It starts with an input-output structure. "
        "You should use three basic structures to build the solving process, "
        "including sequences, branches, and loops. "
        "The necessary details should be written in natural languages.\n\n"
        + examples
        + "\n\n"
        "NEW REQUIREMENT:\n"
        + problem_prompt
        + "\n\n"
        "Please understand the requirement and write a rough solving process. "
        "Start with the Input and Output structure. "
        "Use sequence, branch, and loop structures where appropriate. "
        "Do not write the final C++ code."
    )


def build_code_prompt(problem_prompt, scot):
    examples = ""

    for ex in SCOT_EXAMPLES:
        examples += (
            "\n\nEXAMPLE REQUIREMENT:\n"
            + ex["problem"]
            + "\n\n"
            "EXAMPLE SCoT:\n"
            + ex["scot"]
            + "\n\n"
            "EXAMPLE CODE:\n"
            + ex["code"]
        )

    return (
        examples
        + "\n\n"
        "NEW REQUIREMENT:\n"
        + problem_prompt
        + "\n\n"
        "SCoT:\n"
        + scot
        + "\n\n"
        "# Please check the above solving process and write a "
        "C++ code based on it. Note that the solving process may "
        "contain errors.\n\n"
        "The function declaration is already provided. "
        "Return only the function body. "
        "Do not repeat the function declaration. "
        "Do not use markdown code fences."
    )


print("=" * 70)
print("CODE-EVOL-200 + ORIGINAL-STYLE SCoT — MBCPP")
print("=" * 70)

print("\nSCoT settings:")
print(f"  SCoTs/problem       : {NUM_SCOTS}")
print(f"  SCoT max tokens     : {MAX_SCOT_TOKENS}")
print(f"  SCoT temperature    : {SCOT_TEMPERATURE}")
print(f"  SCoT top-p          : {SCOT_TOP_P}")
print(f"  Code max tokens     : {MAX_CODE_TOKENS}")
print(f"  Code temperature    : {CODE_TEMPERATURE}")

print("\nLoading MBCPP...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    problems = [json.loads(line) for line in f if line.strip()][:3]

print(f"Problems: {len(problems)}")
print(f"Programs/problem: {NUM_SCOTS}")
print(f"Expected programs: {len(problems) * NUM_SCOTS}")
print("Metric: pass@1")

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

print("Model ready.\n")

results = []

empty_scot_count = 0
empty_code_count = 0
scot_fence_count = 0
code_fence_count = 0
full_function_count = 0


for i, problem in enumerate(problems):

    task_id = problem["task_id"]
    entry_point = problem["entry_point"]
    prompt = problem["prompt"]

    print(f"[{i + 1}/{len(problems)}] {task_id}")

    # ============================================================
    # Generate 20 SCoTs
    # ============================================================

    scot_messages = [
        {
            "role": "user",
            "content": build_scot_prompt(prompt)
        }
    ]

    scot_prompt = tokenizer.apply_chat_template(
        scot_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    scot_inputs = tokenizer(
        scot_prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        scot_output = model.generate(
            **scot_inputs,
            max_new_tokens=MAX_SCOT_TOKENS,
            do_sample=True,
            temperature=SCOT_TEMPERATURE,
            top_p=SCOT_TOP_P,
            num_return_sequences=NUM_SCOTS,
            pad_token_id=tokenizer.eos_token_id
        )

    prompt_len = scot_inputs["input_ids"].shape[1]

    scots = []

    for j in range(NUM_SCOTS):

        generated = scot_output[j][prompt_len:]

        scot = tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        if "```" in scot:
            scot_fence_count += 1

        if not scot:
            empty_scot_count += 1

        scots.append(scot)

    del scot_inputs, scot_output
    torch.cuda.empty_cache()

    print(f"  {len(scots)} SCoTs generated.")

    # ============================================================
    # Generate one code for each SCoT
    # ============================================================

    for j, scot in enumerate(scots):

        code_messages = [
            {
                "role": "user",
                "content": build_code_prompt(
                    prompt,
                    scot
                )
            }
        ]

        code_prompt = tokenizer.apply_chat_template(
            code_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        code_inputs = tokenizer(
            code_prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():

            code_output = model.generate(
                **code_inputs,
                max_new_tokens=MAX_CODE_TOKENS,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )

        code_generated = code_output[
            0
        ][code_inputs["input_ids"].shape[1]:]

        raw_completion = tokenizer.decode(
            code_generated,
            skip_special_tokens=True
        ).strip()

        if "```" in raw_completion:
            code_fence_count += 1

        if re.search(
            rf"b{re.escape(entry_point)}s*([^{{}}]*)s*{{",
            raw_completion,
            flags=re.S
        ):
            full_function_count += 1

        completion = clean_completion(
            raw_completion,
            entry_point
        )

        if not completion:
            empty_code_count += 1

        results.append({
            "task_id": task_id,
            "language": problem["language"],
            "scot": scot,
            "completion": completion
        })

        del code_inputs, code_output, code_generated
        torch.cuda.empty_cache()

    print(f"  {NUM_SCOTS} codes generated.")


# ==============================================================
# SAVE
# ==============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")


print("\n" + "=" * 70)
print("ORIGINAL-STYLE SCoT GENERATION COMPLETE")
print("=" * 70)

print(f"Problems              : {len(problems)}")
print(f"SCoTs/problem         : {NUM_SCOTS}")
print(f"Programs/problem      : {NUM_SCOTS}")
print(f"Total programs        : {len(results)}")
print(f"SCoT markdown cases   : {scot_fence_count}")
print(f"Code markdown cases   : {code_fence_count}")
print(f"Full functions       : {full_function_count}")
print(f"Empty SCoTs           : {empty_scot_count}")
print(f"Empty completions     : {empty_code_count}")
print(f"Saved                 : {OUTPUT_FILE}")
