# SY Nodes

## SY Krea 2 One-Click

SY Krea 2 One-Click runs a complete Krea 2 text-to-image pass inside one ComfyUI node. Pick a model, type a prompt, connect the image output, and queue the graph.

The node uses ComfyUI's own loaders, conditioning, sampler, and decoder. It adds no pip packages.

## Install

ComfyUI Manager support will use the pack name `SY Nodes`.

For a manual install:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/stableyogiofficial/sy-nodes.git
```

Restart ComfyUI after installing. The node appears under `Stable Yogi` as `SY Krea 2 One-Click`.

The included public workflow is:

```text
workflows/StableYogi-Krea2-OneClick-W09.json
```

## Model files

Put each file in the matching ComfyUI folder.

| File | Folder |
|---|---|
| Krea 2 diffusion model | `ComfyUI/models/diffusion_models/` |
| Qwen3-VL text encoder | `ComfyUI/models/text_encoders/` |
| Qwen Image or Wan 2.1 VAE | `ComfyUI/models/vae/` |
| Optional LoRAs | `ComfyUI/models/loras/` |

Safetensors models load through core ComfyUI. GGUF models need [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

## First image

1. Add `SY Krea 2 One-Click`.
2. Set `model`.
3. Enter a `prompt`.
4. Connect `image` to `Save Image`.
5. Queue the graph.

The remaining inputs have working defaults.

## Inputs

| Input | What it controls |
|---|---|
| `model` | The Krea 2 diffusion model. Krea-like filenames sort first, but the list does not hide other files. |
| `prompt` | The text to generate. Empty text is accepted. |
| `negative` | Text to avoid. When it is empty, CFG 1 or below uses zeroed conditioning. Above CFG 1 the node uses an encoded empty string to prevent black GGUF frames. |
| `quality` | A paired steps and CFG preset: Fast is 8 and 1, Best is 16 and 2, Max is 20 and 3, and Custom reads the custom fields. |
| `size` | Eight presets or Custom. |
| `seed` | The sampling seed. The same model, settings, and seed reproduce the same image. |
| `detail_pass` | Optional second sampling pass. Light uses 0.25 denoise. Full uses 0.40. |
| `lora_1` | First optional LoRA. |
| `lora_1_strength` | Model and text-encoder strength for `lora_1`. A value of 0 skips the slot. |
| `lora_2` | Second optional LoRA, applied after `lora_1`. |
| `lora_2_strength` | Model and text-encoder strength for `lora_2`. A value of 0 skips the slot. |
| `lora_3` | Third optional LoRA, applied after `lora_2`. |
| `lora_3_strength` | Model and text-encoder strength for `lora_3`. A value of 0 skips the slot. |
| `text_encoder` | Qwen3-VL encoder. Auto prefers the most specific Qwen3-VL filename match. |
| `vae` | Decoder. Auto prefers Qwen Image names and falls back to Wan 2.1 names. |
| `custom_steps` | Step count used when `quality` is Custom. |
| `custom_cfg` | CFG used when `quality` is Custom. |
| `custom_width` | Width used when `size` is Custom. The node rounds it down to a multiple of 8. |
| `custom_height` | Height used when `size` is Custom. The node rounds it down to a multiple of 8. |
| `batch_size` | Images generated in one batch, from 1 to 16. |
| `shift` | AuraFlow sampling shift. The measured default is 2.95. |
| `sampler_name` | ComfyUI sampler. The default is `euler`. |
| `scheduler` | ComfyUI scheduler. The default is `simple`. |
| `weight_dtype` | Safetensors loader override. `default` respects the file. GGUF ignores this field because its quantisation is stored in the file. |

## Size presets

Eight presets cover portrait, square, and landscape work.

| Preset | Pixels |
|---|---:|
| Portrait 2:3 | 960 x 1440 |
| Portrait 2:3 big | 1368 x 2048 |
| Portrait 3:4 | 1152 x 1536 |
| Square 1:1 | 1024 x 1024 |
| Square 1:1 big | 1536 x 1536 |
| Landscape 3:2 | 1440 x 960 |
| Landscape 3:2 big | 2048 x 1368 |
| Landscape 16:9 | 1536 x 864 |

## Outputs

| Output | Use |
|---|---|
| `image` | Decoded image |
| `model` | Model after LoRAs and AuraFlow shift |
| `clip` | Loaded text encoder |
| `vae` | Loaded VAE |
| `positive` | Encoded prompt |
| `negative` | Encoded or zeroed negative conditioning |
| `latent` | Sampled latent before decode |

The extra outputs let a face pass or another sampler reuse the same loaded weights. The node does not save files or add metadata. The connected output node controls file writing.

## Auto detection

Auto detection uses filename fragments in a fixed priority order. If it finds no match, the error lists the fragments it tried and the files it found. Select the correct file manually when your filename does not match those fragments.

## Known limits

- The cache holds one diffusion model, one text encoder, and one VAE. Switching files replaces the cached entry.
- GGUF models require ComfyUI-GGUF.
- NVFP4 needs Blackwell hardware. INT4-ConvRot needs a recent ComfyUI build.
- Custom dimensions are limited to 8192 pixels per side and are rounded down to a multiple of 8.
- The node has no custom browser JavaScript or node-specific buttons.
- Tested on ComfyUI 0.32.0 with Python 3.10.11.

## SY Inpaint Crop and SY Inpaint Stitch

These two nodes make small inpaint fixes sharp. `SY Inpaint Crop` cuts the masked part out of the picture with some room around it and scales the cut-out up. The sampler then draws it at full size. `SY Inpaint Stitch` scales the result back down and blends it into the original picture through the mask. Pixels outside the mask stay unchanged.

1. Connect your picture and its mask to `SY Inpaint Crop`.
2. Encode `cropped_image` and `cropped_mask`, sample, and decode as usual.
3. Connect `stitch` and the decoded image to `SY Inpaint Stitch`.

| Input | What it controls |
|---|---|
| `context` | Room around the masked part, as a multiple of its box. Default 1.5. Values below 1.5 can make the model draw a second face or body in the cut-out. |
| `size` | Long side of the cut-out the sampler draws. Default 1024. |
| `min_box` | Smallest cut-out on the original picture. Default 256. |

## Licence

MIT.
