import json
import re

INPUT = "results/codeevol_200_mbcpp_predictions.jsonl"
OUTPUT = "results/codeevol_200_mbcpp_predictions_fixed.jsonl"
PROBLEMS = "external/mxeval/data/mbxp/mbcpp_release_v1.2.jsonl"


def strip_fences(text):
    text = text.strip()

    text = re.sub(r"^```(?:cpp|c\+\+)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)

    return text.strip()


def extract_function_body(code, entry_point):
    code = strip_fences(code)

    # Find the generated definition of the expected function.
    pattern = rf"\b{re.escape(entry_point)}\s*\([^;{{}}]*\)\s*\{{"
    match = re.search(pattern, code, flags=re.S)

    if not match:
        return code

    open_brace = code.find("{", match.start())
    depth = 0

    for i in range(open_brace, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1

            if depth == 0:
                # Return only the body.
                return code[open_brace + 1:i].strip()

    return code


problems = {}

with open(PROBLEMS, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            p = json.loads(line)
            problems[p["task_id"]] = p


fixed = 0
fences = 0

with open(INPUT, encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

with open(OUTPUT, "w", encoding="utf-8") as f:
    for sample in samples:
        task_id = sample["task_id"]
        entry_point = problems[task_id]["entry_point"]

        original = sample["completion"]

        if "```" in original:
            fences += 1

        completion = extract_function_body(original, entry_point)

        if completion != original:
            fixed += 1

        sample["completion"] = completion

        f.write(json.dumps(sample) + "\n")

print("=" * 60)
print("MBCPP COMPLETION FORMAT FIX")
print("=" * 60)
print("Input records :", len(samples))
print("Fixed         :", fixed)
print("Markdown cases:", fences)
print("Output        :", OUTPUT)
