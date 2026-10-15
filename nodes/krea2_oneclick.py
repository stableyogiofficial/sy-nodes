"""SY Krea2 One-Click - the whole Krea2 text-to-image pipeline in a single node."""

import torch

import comfy.samplers
import folder_paths
import nodes


# --- Tuned defaults for the Krea2 / Flux-DiT family -------------------------

QUALITY_PRESETS = {
    "Fast (8 steps)": {"steps": 8, "cfg": 1.0},
    "Best (16 steps)": {"steps": 16, "cfg": 2.0},
    "Max (20 steps)": {"steps": 20, "cfg": 3.0},
    "Custom": None,
}

SIZE_PRESETS = {
    "Portrait 2:3 (960x1440)": (960, 1440),
    "Portrait 2:3 big (1368x2048)": (1368, 2048),
    "Portrait 3:4 (1152x1536)": (1152, 1536),
    "Square 1:1 (1024x1024)": (1024, 1024),
    "Square 1:1 big (1536x1536)": (1536, 1536),
    "Landscape 3:2 (1440x960)": (1440, 960),
    "Landscape 3:2 big (2048x1368)": (2048, 1368),
    "Landscape 16:9 (1536x864)": (1536, 864),
    "Custom": None,
}

DETAIL_PASS = {"Off": 0.0, "Light": 0.25, "Full": 0.40}

# Hints are ordered best-first and searched in that order, so a loose word like
# "flux" never beats an exact one like "krea". Surface likely names first in the
# dropdowns, but never hide anything the user actually has.
# ponytail: substring match on the filename; swap for a safetensors header probe
# if users start shipping Krea2 files with unrelated names.
MODEL_HINTS = ("krea", "convrot", "muse", "flux")
CLIP_HINTS = ("qwen3vl_4b_fp8", "qwen3vl_4b", "qwen3vl", "qwen3-vl", "qwen3_vl")
VAE_HINTS = ("qwen_image_vae", "qwenimage", "qwen-image", "wan_2.1_vae", "wan2_1_vae", "wan2.1_vae")

AUTO = "auto"
NONE = "None"


def _rank(names, hints):
    """Group by which hint matched, best hint first; unmatched names last."""
    remaining = list(names)
    ranked = []
    for hint in hints:
        tier = [n for n in remaining if hint in n.lower()]
        ranked.extend(tier)
        remaining = [n for n in remaining if n not in tier]
    return ranked + remaining


def _pick_auto(names, hints, what):
    for hint in hints:
        for n in names:
            if hint in n.lower():
                return n
    raise RuntimeError(
        f"SY Krea2: could not auto-find a {what}. Looked for {hints} in: "
        f"{names[:10] if names else 'an empty folder'}. "
        f"Set the {what} dropdown by hand."
    )


def _listing(kind, hints):
    names = folder_paths.get_filename_list(kind)
    return _rank(names, hints) if names else [f"(no {kind} found)"]


def _gguf_names():
    """GGUF checkpoints, or nothing when ComfyUI-GGUF is not installed.

    ComfyUI-GGUF registers its own folder key rather than adding .gguf to
    diffusion_models, so the core loader's list never contains them.
    """
    try:
        return list(folder_paths.get_filename_list("unet_gguf"))
    except Exception:
        return []


def _model_listing():
    names = list(folder_paths.get_filename_list("diffusion_models")) + _gguf_names()
    return _rank(names, MODEL_HINTS) if names else ["(no diffusion_models found)"]


def _load_diffusion_model(name, weight_dtype):
    if not name.lower().endswith(".gguf"):
        return nodes.UNETLoader().load_unet(name, weight_dtype)[0]
    loader = nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
    if loader is None:
        raise RuntimeError(
            "SY Krea2: '" + name + "' is a GGUF file, which needs the ComfyUI-GGUF "
            "node pack (by city96). Install it from ComfyUI Manager and restart, "
            "or pick a .safetensors model instead."
        )
    # GGUF carries its own quantisation, so weight_dtype does not apply.
    return getattr(loader(), loader.FUNCTION)(name)[0]


# --- Loader cache ----------------------------------------------------------
# A reference here keeps the weights resident, so a seed change re-samples
# instead of re-reading 12 GB from disk.
# ponytail: one slot per kind; make it a small LRU if A/B-ing two checkpoints
# in one session becomes a normal workflow.
_CACHE = {}


def _cached(kind, key, build):
    slot = _CACHE.get(kind)
    if slot is not None and slot[0] == key:
        return slot[1]
    value = build()
    _CACHE[kind] = (key, value)
    return value


class SYKrea2OneClick:
    CATEGORY = "Stable Yogi"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("image", "model", "clip", "vae", "positive", "negative", "latent")
    DESCRIPTION = (
        "Krea2 text-to-image in one node: loads the checkpoint, text encoder and "
        "VAE, applies up to 3 LoRAs, samples, optionally runs a detail pass, and "
        "returns the image."
    )

    @classmethod
    def INPUT_TYPES(cls):
        loras = [NONE] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": (_model_listing(), {"tooltip": "Krea 2 diffusion model. Safetensors works in core ComfyUI; GGUF needs ComfyUI-GGUF."}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Describe the image to generate. Empty text is allowed for testing."}),
                "negative": ("STRING", {"multiline": True, "default": "", "tooltip": "Describe what to avoid. Empty is handled safely at every CFG value."}),
                "quality": (list(QUALITY_PRESETS), {"default": "Best (16 steps)", "tooltip": "Sets steps and CFG together. Choose Custom to use custom_steps and custom_cfg."}),
                "size": (list(SIZE_PRESETS), {"default": "Portrait 2:3 (960x1440)", "tooltip": "Output dimensions. Choose Custom to use custom_width and custom_height."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "Random seed. The same seed and settings reproduce the same image."}),
                "detail_pass": (list(DETAIL_PASS), {"default": "Off", "tooltip": "Optional second sampling pass: Light uses 0.25 denoise; Full uses 0.40."}),
            },
            "optional": {
                "lora_1": (loras, {"default": NONE, "tooltip": "First optional LoRA, applied to both model and text encoder."}),
                "lora_1_strength": ("FLOAT", {"default": 0.8, "min": -4.0, "max": 4.0, "step": 0.05, "tooltip": "Strength for LoRA 1. Set to 0 to skip loading this slot."}),
                "lora_2": (loras, {"default": NONE, "tooltip": "Second optional LoRA, applied after LoRA 1."}),
                "lora_2_strength": ("FLOAT", {"default": 0.8, "min": -4.0, "max": 4.0, "step": 0.05, "tooltip": "Strength for LoRA 2. Set to 0 to skip loading this slot."}),
                "lora_3": (loras, {"default": NONE, "tooltip": "Third optional LoRA, applied after LoRA 2."}),
                "lora_3_strength": ("FLOAT", {"default": 0.8, "min": -4.0, "max": 4.0, "step": 0.05, "tooltip": "Strength for LoRA 3. Set to 0 to skip loading this slot."}),
                "text_encoder": ([AUTO] + _listing("text_encoders", CLIP_HINTS), {"default": AUTO, "tooltip": "Qwen3-VL text encoder. Auto picks the best filename match; select a file to override."}),
                "vae": ([AUTO] + _listing("vae", VAE_HINTS), {"default": AUTO, "tooltip": "Krea 2 compatible VAE. Auto prefers Qwen Image and falls back to Wan 2.1 names."}),
                "custom_steps": ("INT", {"default": 16, "min": 1, "max": 200, "tooltip": "Sampling steps used only when quality is Custom."}),
                "custom_cfg": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "CFG used only when quality is Custom."}),
                "custom_width": ("INT", {"default": 960, "min": 64, "max": 8192, "step": 16, "tooltip": "Width used only for Custom size. It is rounded down to a multiple of 8."}),
                "custom_height": ("INT", {"default": 1440, "min": 64, "max": 8192, "step": 16, "tooltip": "Height used only for Custom size. It is rounded down to a multiple of 8."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "tooltip": "Number of images generated in one batch."}),
                "shift": ("FLOAT", {"default": 2.95, "min": 0.0, "max": 20.0, "step": 0.01, "tooltip": "AuraFlow model-sampling shift. The measured Krea 2 default is 2.95."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "Sampling algorithm. Euler is the measured default for Krea 2."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "Noise schedule. Simple is the measured default for Krea 2."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default", "tooltip": "Loader dtype override for safetensors. Default respects the file; GGUF ignores this setting."}),
            },
        }

    def run(
        self,
        model,
        prompt,
        negative,
        quality,
        size,
        seed,
        detail_pass,
        lora_1=NONE,
        lora_1_strength=0.8,
        lora_2=NONE,
        lora_2_strength=0.8,
        lora_3=NONE,
        lora_3_strength=0.8,
        text_encoder=AUTO,
        vae=AUTO,
        custom_steps=16,
        custom_cfg=2.0,
        custom_width=960,
        custom_height=1440,
        batch_size=1,
        shift=2.95,
        sampler_name="euler",
        scheduler="simple",
        weight_dtype="default",
    ):
        steps, cfg = self._quality(quality, custom_steps, custom_cfg)
        width, height = self._size(size, custom_width, custom_height)

        clip_name = text_encoder
        if clip_name == AUTO:
            clip_name = _pick_auto(folder_paths.get_filename_list("text_encoders"), CLIP_HINTS, "text encoder")
        vae_name = vae
        if vae_name == AUTO:
            vae_name = _pick_auto(folder_paths.get_filename_list("vae"), VAE_HINTS, "VAE")

        unet = _cached("unet", (model, weight_dtype), lambda: _load_diffusion_model(model, weight_dtype))
        clip = _cached("clip", clip_name, lambda: nodes.CLIPLoader().load_clip(clip_name, "krea2", "default")[0])
        vae_obj = _cached("vae", vae_name, lambda: nodes.VAELoader().load_vae(vae_name)[0])

        requested = [
            (lora_1, lora_1_strength),
            (lora_2, lora_2_strength),
            (lora_3, lora_3_strength),
        ]
        unet, clip = self._apply_loras(unet, clip, requested)

        unet = nodes.NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]().patch_aura(unet, shift)[0]

        encoder = nodes.CLIPTextEncode()
        positive = encoder.encode(clip, prompt)[0]
        if negative.strip():
            negative_cond = encoder.encode(clip, negative)[0]
        elif cfg <= 1.0:
            # At CFG 1 the negative branch is mathematically ignored, so the
            # cheap zeroed conditioning is free.
            negative_cond = nodes.ConditioningZeroOut().zero_out(positive)[0]
        else:
            # Above CFG 1 the negative is actually subtracted, and a zeroed
            # conditioning decodes to a pure black frame on GGUF checkpoints.
            # An encoded empty string is a real conditioning and is safe on
            # every format. Measured 2026-09-12 on a Q8_0 file at CFG 2:
            # zeroed -> std 0.00 (black), empty string -> a normal image.
            negative_cond = encoder.encode(clip, "")[0]

        latent = {"samples": torch.zeros([batch_size, 16, height // 8, width // 8])}

        sampled = nodes.common_ksampler(
            unet, seed, steps, cfg, sampler_name, scheduler,
            positive, negative_cond, latent, denoise=1.0,
        )[0]

        boost = DETAIL_PASS[detail_pass]
        if boost > 0.0:
            sampled = nodes.common_ksampler(
                unet, seed + 1, steps, cfg, sampler_name, scheduler,
                positive, negative_cond, sampled, denoise=boost,
            )[0]

        image = nodes.VAEDecode().decode(vae_obj, sampled)[0]
        # The model is returned post-LoRA and post-shift so a downstream detail or
        # face pass samples with exactly the same weights this image came from.
        return (image, unet, clip, vae_obj, positive, negative_cond, sampled)

    @staticmethod
    def _quality(quality, custom_steps, custom_cfg):
        preset = QUALITY_PRESETS.get(quality)
        if preset is None:
            return custom_steps, custom_cfg
        return preset["steps"], preset["cfg"]

    @staticmethod
    def _size(size, custom_width, custom_height):
        preset = SIZE_PRESETS.get(size)
        width, height = preset if preset else (custom_width, custom_height)
        # The VAE works in units of 8px; a stray odd value would crash the decode.
        return max(64, width // 8 * 8), max(64, height // 8 * 8)

    @staticmethod
    def _apply_loras(unet, clip, requested):
        loader = nodes.LoraLoader()
        for name, strength in requested:
            if name == NONE or not name or strength == 0.0:
                continue
            unet, clip = loader.load_lora(unet, clip, name, strength, strength)
        return unet, clip


NODE_CLASS_MAPPINGS = {"SYKrea2OneClick": SYKrea2OneClick}
NODE_DISPLAY_NAME_MAPPINGS = {"SYKrea2OneClick": "SY Krea2 One-Click"}
