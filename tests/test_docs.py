"""Documentation and package contract for SY Krea 2 One-Click."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SPEC = ROOT / "SPEC.md"

INPUTS = [
    "model",
    "prompt",
    "negative",
    "quality",
    "size",
    "seed",
    "detail_pass",
    "lora_1",
    "lora_1_strength",
    "lora_2",
    "lora_2_strength",
    "lora_3",
    "lora_3_strength",
    "text_encoder",
    "vae",
    "custom_steps",
    "custom_cfg",
    "custom_width",
    "custom_height",
    "batch_size",
    "shift",
    "sampler_name",
    "scheduler",
    "weight_dtype",
]


def main():
    failures = []
    text = README.read_text(encoding="utf-8")

    for name in INPUTS:
        if f"`{name}`" not in text:
            failures.append(f"README does not name input {name}")

    required_truths = {
        "manual install command": "git clone https://github.com/stableyogiofficial/sy-nodes.git",
        "Qwen Image priority": "prefers Qwen Image",
        "Wan fallback": "falls back to Wan 2.1",
        "CFG 1 negative rule": "CFG 1 or below",
        "CFG above 1 negative rule": "encoded empty string",
        "eight size presets": "Eight presets",
        "known limits section": "## Known limits",
        "GGUF dependency": "ComfyUI-GGUF",
        "no pip dependencies": "No pip packages",
    }
    for label, snippet in required_truths.items():
        if snippet.lower() not in text.lower():
            failures.append(f"README missing {label}: {snippet!r}")

    if not SPEC.exists():
        failures.append("SPEC.md is missing")
    else:
        spec = SPEC.read_text(encoding="utf-8")
        for heading in (
            "# SY Krea 2 One-Click specification",
            "## Purpose",
            "## Inputs",
            "## Outputs",
            "## Runtime order",
            "## Limits",
        ):
            if heading not in spec:
                failures.append(f"SPEC.md missing {heading}")

    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files"], text=True, encoding="utf-8"
    ).splitlines()
    dirty_artifacts = [
        name
        for name in tracked
        if name.endswith((".bak", ".pyc")) or "__pycache__" in name.split("/")
    ]
    if dirty_artifacts:
        failures.append("tracked build artifacts: " + ", ".join(dirty_artifacts))

    if failures:
        print("RED documentation/package contract")
        for failure in failures:
            print(" - " + failure)
        return 1

    print("GREEN README covers every input and corrected behavior")
    print("GREEN SPEC.md carries purpose, inputs, outputs, runtime order, and limits")
    print("GREEN no .bak, .pyc, or __pycache__ files are tracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())

