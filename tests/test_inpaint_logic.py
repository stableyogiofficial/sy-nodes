"""Fast logic checks for SY Inpaint Crop + Stitch (torch only, no ComfyUI needed).

Run:  <python with torch> tests/test_inpaint_logic.py        -> ends "ALL PASS <n>"
      ... --known-bad                                        -> must FAIL (proves the checks bite)
"""

import argparse
import importlib.util
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "inpaint_crop_stitch.py"
spec = importlib.util.spec_from_file_location("sy_inpaint_under_test", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

passed = 0


def check(cond, msg):
    global passed
    if not cond:
        raise AssertionError(msg)
    passed += 1


def picture(b=1, h=1440, w=960):
    torch.manual_seed(0)
    return torch.rand(b, h, w, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-bad", action="store_true")
    args = ap.parse_args()
    if args.known_bad:
        mod.MULTIPLE = 10

    crop, stitch = mod.SYInpaintCrop(), mod.SYInpaintStitch()

    # 1. small hand-sized mark in a big picture -> box with room, crop scaled to long side 1024, /16
    img = picture()
    mask = torch.zeros(1, 1440, 960)
    mask[:, 1100:1200, 300:400] = 1.0
    st, cimg, cmask = crop.crop(img, mask, 1.5, 1024, 256)
    x0, y0, x1, y1 = st["box"]
    check((x1 - x0, y1 - y0) == (256, 256), f"box should be min_box 256 square, got {(x1 - x0, y1 - y0)}")
    check(cimg.shape == (1, 1024, 1024, 3), f"crop shape {tuple(cimg.shape)}")
    check(cmask.shape == (1, 1024, 1024), f"crop mask shape {tuple(cmask.shape)}")
    check(cimg.shape[1] % 16 == 0 and cimg.shape[2] % 16 == 0, "crop not a multiple of 16")
    check(x0 <= 300 and x1 >= 400 and y0 <= 1100 and y1 >= 1200, "box does not hold the mark")

    # 2. stitch with the untouched crop -> outside the mask is pixel for pixel the original
    out = stitch.stitch(st, cimg)[0]
    outside = mask[0] == 0
    check(torch.equal(out[0][outside], img[0][outside]), "pixels outside the mask changed")
    check(out.shape == img.shape, "stitched size changed")

    # 3. stitch with a white crop -> inside the mask is white
    out = stitch.stitch(st, torch.ones_like(cimg))[0]
    check(torch.allclose(out[0, 1120:1180, 320:380], torch.ones(60, 60, 3), atol=1e-4), "inside the mask not replaced")

    # 4. box near the edge stays inside the picture
    mask = torch.zeros(1, 1440, 960)
    mask[:, 0:40, 920:960] = 1.0
    st, cimg, _ = crop.crop(img, mask, 2.0, 1024, 256)
    x0, y0, x1, y1 = st["box"]
    check(x0 >= 0 and y0 >= 0 and x1 <= 960 and y1 <= 1440, f"box left the picture {st['box']}")
    check(stitch.stitch(st, cimg)[0].shape == img.shape, "edge stitch size")

    # 5. empty mask -> whole picture, graph still runs, nothing changes
    st, cimg, _ = crop.crop(img, torch.zeros(1, 1440, 960), 1.5, 1024, 256)
    check(st["box"] == (0, 0, 960, 1440), f"empty mask box {st['box']}")
    check(cimg.shape == (1, 1024, 688, 3), f"empty mask crop {tuple(cimg.shape)}")
    out = stitch.stitch(st, cimg)[0]
    check(torch.equal(out, img), "empty mask changed the picture")

    # 6. big mask bigger than size -> scaled down, still /16
    mask = torch.zeros(1, 1440, 960)
    mask[:, 100:1400, 50:900] = 1.0
    st, cimg, _ = crop.crop(img, mask, 1.5, 1024, 256)
    check(cimg.shape[1] == 1024 and cimg.shape[2] % 16 == 0, f"big mask crop {tuple(cimg.shape)}")

    # 7. batch of 4 with one mask (2-D mask too)
    img4 = picture(4)
    mask2d = torch.zeros(1440, 960)
    mask2d[700:800, 400:500] = 1.0
    st, cimg, cmask = crop.crop(img4, mask2d, 1.5, 1024, 256)
    check(cimg.shape[0] == 4 and cmask.shape[0] == 4, "batch of 4 lost")
    check(stitch.stitch(st, cimg)[0].shape == img4.shape, "batch stitch size")

    # 8. mask at another size than the picture is scaled to fit
    st, _, _ = crop.crop(img, torch.ones(1, 720, 480), 1.5, 1024, 256)
    check(st["mask"].shape[1:] == (1440, 960), "mask was not scaled to the picture")

    # 9. registry names
    check(set(mod.NODE_CLASS_MAPPINGS) == {"SYInpaintCrop", "SYInpaintStitch"}, "class names drifted")

    print("ALL PASS", passed)


if __name__ == "__main__":
    main()
