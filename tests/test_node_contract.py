"""Contract checks for SY Krea 2 One-Click running on the isolated bench."""

import json
import os
import sys
import urllib.error
import urllib.request


OBJECT_INFO_URL = os.environ.get("SY_COMFY_OBJECT_INFO_URL")


def load_node_info():
    if not OBJECT_INFO_URL:
        raise RuntimeError("set SY_COMFY_OBJECT_INFO_URL to the live node object-info URL")
    with urllib.request.urlopen(OBJECT_INFO_URL, timeout=10) as response:
        payload = json.load(response)
    return payload["SYKrea2OneClick"]


def main():
    try:
        info = load_node_info()
    except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
        print(
            "RED configured ComfyUI endpoint unavailable. " + str(error)
        )
        return 1
    missing = []
    for section in ("required", "optional"):
        for name, spec in info["input"][section].items():
            options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if not str(options.get("tooltip", "")).strip():
                missing.append(f"{section}.{name}")
    if missing:
        print("RED missing input tooltips: " + ", ".join(missing))
        return 1
    print("GREEN all required and optional inputs have non-empty tooltips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
