# SY Krea 2 One-Click specification

## Purpose

Reduce a normal Krea 2 text-to-image graph to one node while keeping the loaded model, encoder, VAE, conditionings, and sampled latent available for downstream work.

The implementation calls ComfyUI's own node classes and sampling function. It has no third-party Python dependency. GGUF support is optional and appears when ComfyUI-GGUF is installed.

## Inputs

The required inputs are `model`, `prompt`, `negative`, `quality`, `size`, `seed`, and `detail_pass`.

Optional inputs provide three ordered LoRA slots with independent strengths, manual text-encoder and VAE selection, custom steps and CFG, custom width and height, batch size, AuraFlow shift, sampler, scheduler, and loader dtype.

Every input must have a non-empty ComfyUI tooltip. A LoRA slot is skipped when its name is `None`, its name is empty, or its strength is zero.

## Outputs

The node returns seven values in this order:

1. `image`
2. `model`
3. `clip`
4. `vae`
5. `positive`
6. `negative`
7. `latent`

The model output is the same patched model used for the first sampling pass.

## Runtime order

1. Resolve quality and size presets.
2. Resolve automatic text-encoder and VAE choices.
3. Load or reuse the diffusion model, text encoder, and VAE.
4. Apply up to three LoRAs in slot order.
5. Apply the AuraFlow sampling shift.
6. Encode the prompt.
7. Build negative conditioning. Empty negative uses zeroed conditioning at CFG 1 or below and encoded empty text above CFG 1.
8. Allocate a 16-channel latent.
9. Sample once.
10. Run the optional Light or Full detail pass at `seed + 1`.
11. Decode the sampled latent.
12. Return the image and six reusable pipeline values.

## Cache behavior

The module cache keeps one value for each of the diffusion model, text encoder, and VAE. The key includes the diffusion filename and loader dtype for the model entry. Repeating a run with a new seed reuses all three entries. Changing a filename replaces the matching entry.

## Automatic selection

Text-encoder priority starts with `qwen3vl_4b_fp8` and then tries broader Qwen3-VL fragments.

VAE priority starts with `qwen_image_vae`. Wan 2.1 fragments are the fallback.

The model dropdown ranks Krea, ConvRot, Muse, and Flux-like names first without filtering the rest.

## Limits

- One cached file per loader kind. A small LRU is the upgrade path if two-model comparisons become common.
- Filename fragments drive automatic selection. Header inspection is the upgrade path if unrelated files begin using the same fragments.
- Custom width and height are clamped to at least 64 and rounded down to a multiple of 8.
- GGUF loading is unavailable when ComfyUI-GGUF is absent.
- The node has no custom browser JavaScript.
- File writing and metadata belong to downstream output nodes.

## Verification

Run the tooltip contract against a live bench:

```text
python tests/test_node_contract.py
```

Run logic and documentation checks:

```text
python tests/test_node_logic.py
python tests/test_docs.py
```

Run the public test suite from the repository root:

```text
python -m pytest tests
```
