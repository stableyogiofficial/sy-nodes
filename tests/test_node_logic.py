"""Fast logic checks for the real SY Krea 2 One-Click module."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys


COMFY_ROOT_VALUE = os.environ.get("COMFYUI_ROOT")
MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "krea2_oneclick.py"


def load_module():
    if not COMFY_ROOT_VALUE:
        raise RuntimeError("set COMFYUI_ROOT to the local ComfyUI folder")
    comfy_root = Path(COMFY_ROOT_VALUE).expanduser().resolve()
    sys.path.insert(0, str(comfy_root))
    spec = importlib.util.spec_from_file_location("sy_krea2_oneclick_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--known-bad", action="store_true")
    args = parser.parse_args()

    module = load_module()
    if args.known_bad:
        module.QUALITY_PRESETS["Fast (8 steps)"]["steps"] = 9

    check(module.SYKrea2OneClick._quality("Fast (8 steps)", 99, 9.9) == (8, 1.0), "Fast preset drifted")
    check(module.SYKrea2OneClick._quality("Best (16 steps)", 99, 9.9) == (16, 2.0), "Best preset drifted")
    check(module.SYKrea2OneClick._quality("Max (20 steps)", 99, 9.9) == (20, 3.0), "Max preset drifted")
    check(module.SYKrea2OneClick._quality("Custom", 7, 1.7) == (7, 1.7), "Custom quality ignored fields")

    expected_sizes = {
        "Portrait 2:3 (960x1440)": (960, 1440),
        "Portrait 2:3 big (1368x2048)": (1368, 2048),
        "Portrait 3:4 (1152x1536)": (1152, 1536),
        "Square 1:1 (1024x1024)": (1024, 1024),
        "Square 1:1 big (1536x1536)": (1536, 1536),
        "Landscape 3:2 (1440x960)": (1440, 960),
        "Landscape 3:2 big (2048x1368)": (2048, 1368),
        "Landscape 16:9 (1536x864)": (1536, 864),
    }
    for name, size in expected_sizes.items():
        check(module.SYKrea2OneClick._size(name, 1, 1) == size, f"size preset drifted: {name}")
    check(module.SYKrea2OneClick._size("Custom", 963, 1443) == (960, 1440), "odd custom size did not round down")
    check(module.SYKrea2OneClick._size("Custom", 1, 7) == (64, 64), "minimum custom size was not enforced")

    names = ["generic-qwen3vl.safetensors", "qwen3vl_4b_fp8_scaled.safetensors"]
    picked = module._pick_auto(names, module.CLIP_HINTS, "text encoder")
    check(picked == "qwen3vl_4b_fp8_scaled.safetensors", "auto selection ignored priority order")

    for what, hints in (("text encoder", module.CLIP_HINTS), ("VAE", module.VAE_HINTS)):
        try:
            module._pick_auto(["unrelated.bin"], hints, what)
        except RuntimeError as error:
            message = str(error)
            check(f"could not auto-find a {what}" in message, f"{what} error omits what failed")
            check("unrelated.bin" in message, f"{what} error omits discovered files")
            check(f"Set the {what} dropdown by hand" in message, f"{what} error omits the fix")
        else:
            raise AssertionError(f"{what} auto failure did not raise")

    real_get_filename_list = module.folder_paths.get_filename_list

    def unavailable_gguf(kind):
        if kind == "unet_gguf":
            raise KeyError(kind)
        return real_get_filename_list(kind)

    module.folder_paths.get_filename_list = unavailable_gguf
    try:
        check(module._gguf_names() == [], "missing GGUF pack did not degrade to an empty list")
    finally:
        module.folder_paths.get_filename_list = real_get_filename_list

    gguf_loader = module.nodes.NODE_CLASS_MAPPINGS.pop("UnetLoaderGGUF", None)
    try:
        try:
            module._load_diffusion_model("missing.gguf", "default")
        except RuntimeError as error:
            check("needs the ComfyUI-GGUF node pack" in str(error), "GGUF error omits dependency and fix")
        else:
            raise AssertionError("GGUF model without its loader did not raise")
    finally:
        if gguf_loader is not None:
            module.nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"] = gguf_loader

    module._CACHE.clear()
    builds = []

    def build(value):
        builds.append(value)
        return object()

    first = module._cached("unet", ("model-a", "default"), lambda: build("model-a"))
    second = module._cached("unet", ("model-a", "default"), lambda: build("model-a-repeat"))
    third = module._cached("unet", ("model-b", "default"), lambda: build("model-b"))
    check(first is second, "same cache key rebuilt the model")
    check(third is not second, "new cache key reused the old model")
    check(builds == ["model-a", "model-b"], f"cache build calls wrong: {builds}")

    calls = []

    class FakeLoraLoader:
        def load_lora(self, model, clip, name, model_strength, clip_strength):
            calls.append((name, model_strength, clip_strength))
            return model + f"|{name}", clip + f"|{name}"

    real_loader = module.nodes.LoraLoader
    module.nodes.LoraLoader = FakeLoraLoader
    try:
        model, clip = module.SYKrea2OneClick._apply_loras(
            "model",
            "clip",
            [
                ("first.safetensors", 0.0),
                ("second.safetensors", 0.4),
                ("", 0.8),
                ("third.safetensors", -0.2),
            ],
        )
    finally:
        module.nodes.LoraLoader = real_loader

    check(
        calls == [
            ("second.safetensors", 0.4, 0.4),
            ("third.safetensors", -0.2, -0.2),
        ],
        f"LoRA skip/order/strength contract failed: {calls}",
    )
    check(model.endswith("|second.safetensors|third.safetensors"), "LoRA model order drifted")
    check(clip.endswith("|second.safetensors|third.safetensors"), "LoRA clip order drifted")

    check(module.DETAIL_PASS == {"Off": 0.0, "Light": 0.25, "Full": 0.40}, "detail presets drifted")
    check(len(module.SYKrea2OneClick.RETURN_TYPES) == 7, "node no longer returns seven outputs")
    check(
        module.SYKrea2OneClick.RETURN_NAMES
        == ("image", "model", "clip", "vae", "positive", "negative", "latent"),
        "output names/order drifted",
    )

    print("GREEN quality and size mappings")
    print("GREEN auto priority and actionable encoder/VAE errors")
    print("GREEN same-key cache reuse and changed-key reload")
    print("GREEN LoRA zero skip, order, and paired strengths")
    print("GREEN detail presets and all seven output names")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"RED {error}")
        sys.exit(1)

