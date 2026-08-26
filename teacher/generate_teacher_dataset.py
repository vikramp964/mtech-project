import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = "/workspace/project/models/Qwen2.5-Coder-32B-Instruct"
INPUT_FILE = "/workspace/project/datasets/train.json"
OUTPUT_FILE = "/workspace/project/datasets/code_evol_scot_pilot_v2.json"
FAILED_FILE = "/workspace/project/datasets/code_evol_scot_failed.json"

NUM_SAMPLES = 500
MAX_NEW_TOKENS = 1400

EVOLUTION_TYPES = [
    "Increase Constraints",
    "Increase Complexity",
    "Add Edge Cases",
    "Improve Efficiency Requirements",
    "Improve Code Quality",
    "Generalize the Problem",
]

SYSTEM_PROMPT = """
You are an expert competitive programmer and programming teacher.

Your task is to transform an ORIGINAL programming problem using
Code Evol-Instruct principles and then solve the evolved problem
using Structured Chain-of-Thought (SCoT).

The evolved problem MUST:
- remain directly related to the original problem
- be genuinely more challenging
- preserve the programming language of the original solution
- remain solvable
- contain clear requirements

You MUST choose exactly ONE evolution strategy:

1. Increase Constraints
2. Increase Complexity
3. Add Edge Cases
4. Improve Efficiency Requirements
5. Improve Code Quality
6. Generalize the Problem

For SCoT, provide:
1. concise step-by-step reasoning
2. language-independent pseudocode
3. complete executable final code

Do not include unrelated explanations.

Return ONLY:

Evolution Type:
...

Enhanced Instruction:
...

Reasoning:
1. ...
2. ...
3. ...

Pseudo-code:
...

Final Code:
...
"""

USER_PROMPT = """
ORIGINAL PROGRAMMING TASK:

{instruction}

ADDITIONAL INPUT:

{input}

ORIGINAL SOLUTION:

{output}

Perform ONE Code Evol-Instruct transformation.

Then solve the evolved task using SCoT.

Remember:
- Preserve the original programming language.
- Do not make the task unrelated to the original.
- Make the evolution meaningful.
- The final code must solve the ENHANCED instruction.
"""


def load_model():
    print("=" * 60)
    print("Loading teacher model...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    print("Teacher model loaded successfully!")

    return tokenizer, model


def generate_response(tokenizer, model, example):

    prompt = USER_PROMPT.format(
        instruction=example.get("instruction", ""),
        input=example.get("input", ""),
        output=example.get("output", "")
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.05
        )

    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()


def extract_section(text, start, end=None):

    if start not in text:
        return ""

    section = text.split(start, 1)[1]

    if end and end in section:
        section = section.split(end, 1)[0]

    return section.strip()


def clean_code(code):

    # Remove markdown code fences.
    code = re.sub(r"```[a-zA-Z0-9_+-]*\n?", "", code)
    code = code.replace("```", "")

    # Remove accidental trailing section labels.
    for marker in [
        "\nExplanation:",
        "\n### Explanation:",
        "\nThis code",
        "\nNote:"
    ]:
        if marker in code:
            code = code.split(marker, 1)[0]

    return code.strip()


def parse_response(response):

    evolution = extract_section(
        response,
        "Evolution Type:",
        "Enhanced Instruction:"
    )

    enhanced = extract_section(
        response,
        "Enhanced Instruction:",
        "Reasoning:"
    )

    reasoning = extract_section(
        response,
        "Reasoning:",
        "Pseudo-code:"
    )

    pseudocode = extract_section(
        response,
        "Pseudo-code:",
        "Final Code:"
    )

    final_code = extract_section(
        response,
        "Final Code:"
    )

    final_code = clean_code(final_code)

    return {
        "evolution_type": evolution,
        "enhanced_instruction": enhanced,
        "reasoning": reasoning,
        "pseudocode": pseudocode,
        "final_code": final_code
    }


def validate(record):

    if record["evolution_type"] not in EVOLUTION_TYPES:
        return False, "invalid_evolution_type"

    if len(record["enhanced_instruction"]) < 30:
        return False, "weak_enhanced_instruction"

    if len(record["reasoning"]) < 30:
        return False, "weak_reasoning"

    if len(record["pseudocode"]) < 20:
        return False, "weak_pseudocode"

    if len(record["final_code"]) < 20:
        return False, "weak_final_code"

    return True, "valid"


def main():

    tokenizer, model = load_model()

    with open(INPUT_FILE, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Source samples: {len(dataset)}")
    print(f"Target samples: {NUM_SAMPLES}")

    results = []
    failures = []

    for i, example in enumerate(dataset[:NUM_SAMPLES]):

        print(f"\n[{i + 1}/{NUM_SAMPLES}] Generating...")

        try:

            response = generate_response(
                tokenizer,
                model,
                example
            )

            parsed = parse_response(response)

            record = {
                "original_instruction": example.get(
                    "instruction", ""
                ),
                "original_input": example.get(
                    "input", ""
                ),
                "original_output": example.get(
                    "output", ""
                ),
                **parsed
            }

            valid, reason = validate(record)

            if valid:
                results.append(record)
                print("VALID")

            else:
                failures.append({
                    "original_instruction": example.get(
                        "instruction", ""
                    ),
                    "reason": reason,
                    "raw_response": response
                })
                print(f"REJECTED: {reason}")

        except Exception as e:

            failures.append({
                "original_instruction": example.get(
                    "instruction", ""
                ),
                "reason": "generation_error",
                "error": str(e)
            })

            print(f"ERROR: {e}")

    Path(OUTPUT_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    with open(FAILED_FILE, "w", encoding="utf-8") as f:
        json.dump(
            failures,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("GENERATION COMPLETED")
    print("=" * 60)
    print(f"Valid samples : {len(results)}")
    print(f"Rejected      : {len(failures)}")
    print(f"Saved         : {OUTPUT_FILE}")
    print(f"Failures      : {FAILED_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()



