"""SY Inpaint Crop + SY Inpaint Stitch: make a small painted part at full size, then put it back.

Our own code, written from the idea only (clean room): find the painted area, take a box around it
with some room, scale that box up so the model draws it at full resolution, then scale the result
back down and blend it into the untouched picture with the soft mask.

Only torch and ComfyUI's own types. Nothing to install.
"""

import torch
import torch.nn.functional as F

STITCH_TYPE = "SY_STITCH"
MULTIPLE = 16  # Krea 2: VAE /8 and 2x2 patches -> sizes must divide by 16


def _bbox(mask, threshold=0.01):
    """Box (x0, y0, x1, y1) around every mask pixel above threshold, over the whole batch. None if empty."""
    m = (mask > threshold).any(dim=0)
    ys = torch.nonzero(m.any(dim=1)).flatten()
    xs = torch.nonzero(m.any(dim=0)).flatten()
    if ys.numel() == 0:
        return None
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def _context_box(box, width, height, context, min_side):
    """Grow the box around its centre by `context` (1.0 = tight), at least min_side, kept inside the picture."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw = min(width, max((x1 - x0) * context, min_side))
    bh = min(height, max((y1 - y0) * context, min_side))
    nx0 = int(round(min(max(cx - bw / 2.0, 0), width - bw)))
    ny0 = int(round(min(max(cy - bh / 2.0, 0), height - bh)))
    return nx0, ny0, nx0 + int(round(bw)), ny0 + int(round(bh))


def _target_size(w, h, long_side):
    """Scale (w, h) so the long side is long_side; both sides round to a multiple of 16 (at least 64)."""
    scale = long_side / float(max(w, h))
    tw = max(64, int(round(w * scale / MULTIPLE)) * MULTIPLE)
    th = max(64, int(round(h * scale / MULTIPLE)) * MULTIPLE)
    return tw, th


def _resize_image(img, w, h):
    """IMAGE [B,H,W,C] -> [B,h,w,C]. Bicubic up, area down (no ringing when shrinking)."""
    x = img.movedim(-1, 1)
    if w >= x.shape[3]:
        x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
    else:
        x = F.interpolate(x, size=(h, w), mode="area")
    return x.movedim(1, -1)


def _resize_mask(mask, w, h):
    return F.interpolate(mask.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1).clamp(0, 1)


def _match_batch(t, b):
    return t if t.shape[0] == b else t[:1].expand(b, *t.shape[1:])


class SYInpaintCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Your whole picture."}),
                "mask": ("MASK", {"tooltip": "The painted part (already grown and softened is best)."}),
                "context": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05,
                                      "tooltip": "Room around the painted part, as a factor of its size. 1.5 = half "
                                                 "again on each side in total. More room = the new part matches "
                                                 "the scene better, but gets fewer pixels."}),
                "size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16,
                                 "tooltip": "Long side of the crop the model draws, in pixels. 1024 suits Krea 2."}),
                "min_box": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 16,
                                    "tooltip": "Smallest crop box on the original picture, so a tiny mark still "
                                               "gets some scene around it."}),
            }
        }

    RETURN_TYPES = (STITCH_TYPE, "IMAGE", "MASK")
    RETURN_NAMES = ("stitch", "cropped_image", "cropped_mask")
    FUNCTION = "crop"
    CATEGORY = "Stable Yogi"
    DESCRIPTION = ("Cuts a box around the painted part and scales it up, so the model draws a small hand or face at "
                   "full size. Send 'stitch' to SY Inpaint Stitch to put the result back.")

    def crop(self, image, mask, context, size, min_box):
        b, height, width, _ = image.shape
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[1:] != (height, width):
            mask = _resize_mask(mask, width, height)
        mask = _match_batch(mask, b)

        box = _bbox(mask)
        if box is None:  # nothing painted: the whole picture, so the graph still runs
            box = (0, 0, width, height)
        x0, y0, x1, y1 = _context_box(box, width, height, context, min_box)
        tw, th = _target_size(x1 - x0, y1 - y0, size)

        crop_img = _resize_image(image[:, y0:y1, x0:x1, :], tw, th)
        crop_mask = _resize_mask(mask[:, y0:y1, x0:x1], tw, th)
        stitch = {"original": image, "mask": mask, "box": (x0, y0, x1, y1), "crop_size": (tw, th)}
        return (stitch, crop_img, crop_mask)


class SYInpaintStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitch": (STITCH_TYPE, {"tooltip": "From SY Inpaint Crop."}),
                "inpainted": ("IMAGE", {"tooltip": "The decoded crop from the sampler."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch"
    CATEGORY = "Stable Yogi"
    DESCRIPTION = ("Scales the new crop back down and blends it into your picture through the soft mask. "
                   "Every pixel outside the mask stays your own.")

    def stitch(self, stitch, inpainted):
        original = stitch["original"]
        x0, y0, x1, y1 = stitch["box"]
        b = max(original.shape[0], inpainted.shape[0])
        out = _match_batch(original, b).clone()
        mask = _match_batch(stitch["mask"], b)

        part = _resize_image(inpainted.to(out.device, out.dtype), x1 - x0, y1 - y0)
        part = _match_batch(part, b)
        alpha = mask[:, y0:y1, x0:x1].unsqueeze(-1).to(out.device, out.dtype)
        region = out[:, y0:y1, x0:x1, :]
        out[:, y0:y1, x0:x1, :] = region * (1.0 - alpha) + part * alpha
        return (out,)


NODE_CLASS_MAPPINGS = {"SYInpaintCrop": SYInpaintCrop, "SYInpaintStitch": SYInpaintStitch}
NODE_DISPLAY_NAME_MAPPINGS = {"SYInpaintCrop": "SY Inpaint Crop", "SYInpaintStitch": "SY Inpaint Stitch"}
