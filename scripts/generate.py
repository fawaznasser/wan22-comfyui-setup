#!/usr/bin/env python3
"""
Generate a Wan 2.2 video through a running ComfyUI server (start_wan first).

    python generate.py --prompt "A red fox in the snow" --frames 49 --steps 20
    python generate.py --prompt "She turns and smiles" --image photo.jpg

Uses the API-format workflows in ./workflows and the defaults in ./wan_config.json
(both written by install.py). Standard library only.
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEG_DEFAULT = "blurry, distorted, extra limbs, static, low quality, watermark, text"
LATENT_NODES = {"Wan22ImageToVideoLatent", "EmptyHunyuanLatentVideo", "WanImageToVideo"}
VIDEO_EXT = (".mp4", ".webm", ".mov", ".mkv", ".gif", ".webp")


class ComfyError(Exception):
    pass


def http(server, path, data=None, headers=None, timeout=60):
    req = urllib.request.Request(server + path, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise ComfyError(f"ComfyUI returned {e.code} for {path}:\n{e.read().decode(errors='replace')}")
    except (urllib.error.URLError, OSError) as e:
        raise ComfyError(f"Can't reach ComfyUI at {server} ({getattr(e, 'reason', e)}). "
                         "Start it with start_wan first.")


def load_config():
    p = HERE / "wan_config.json"
    if not p.exists():
        raise ComfyError(f"{p} not found - run install.py first.")
    return json.loads(p.read_text(encoding="utf-8"))


def load_workflow(cfg, key):
    f = Path(cfg["workflows"].get(key, key))
    if not f.is_absolute():
        f = HERE / "workflows" / f
    if not f.exists():
        raise ComfyError(f"Workflow not found: {f}")
    return f, json.loads(f.read_text(encoding="utf-8"))


def queue(server, wf, client_id):
    r = json.loads(http(server, "/prompt", json.dumps({"prompt": wf, "client_id": client_id}).encode(),
                        {"Content-Type": "application/json"}))
    return r["prompt_id"]


def error_text(entry):
    """Human-readable error from a /history entry, or None if it succeeded."""
    status = entry.get("status", {})
    if status.get("status_str") != "error":
        return None
    for kind, msg in status.get("messages", []):
        if kind == "execution_error":
            text = f"Node {msg.get('node_type')} failed: {msg.get('exception_message')}"
            if "out of memory" in str(msg.get("exception_message", "")).lower():
                text += ("\nOut of GPU memory: lower the resolution, then frames, "
                         "then use a smaller model.")
            return text
    return "Generation failed (no details from ComfyUI)."


def save_outputs(server, entry, outdir: Path, name, meta=None):
    files = [f for out in entry.get("outputs", {}).values() for v in out.values() if isinstance(v, list)
             for f in v if isinstance(f, dict) and str(f.get("filename", "")).lower().endswith(VIDEO_EXT)]
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    for k, f in enumerate(files):
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                    "type": f.get("type", "output")})
        dest = outdir / f"{name}_{stamp}{'_' + str(k) if k else ''}{Path(f['filename']).suffix}"
        dest.write_bytes(http(server, f"/view?{q}", timeout=600))
        if meta is not None:
            dest.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        saved.append(dest)
    return saved


def upload_image(server, path: Path):
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
            f'Content-Type: application/octet-stream\r\n\r\n').encode() + path.read_bytes() + \
           (f'\r\n--{b}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue'
            f'\r\n--{b}--\r\n').encode()
    r = json.loads(http(server, "/upload/image", body,
                        {"Content-Type": f"multipart/form-data; boundary={b}"}))
    return f"{r['subfolder']}/{r['name']}" if r.get("subfolder") else r["name"]


def fix_frames(n):
    if (n - 1) % 4 == 0:
        return n
    fixed = max(5, round((n - 1) / 4) * 4 + 1)
    print(f"frames must be 4n+1 - using {fixed} instead of {n}")
    return fixed


def fix_dim(v, mult, label):
    fixed = max(mult, round(v / mult) * mult)
    if fixed != v:
        print(f"{label} must be a multiple of {mult} - using {fixed} instead of {v}")
    return fixed


class VramMonitor(threading.Thread):
    """Polls nvidia-smi for peak used VRAM (whole GPU, MiB)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = None, threading.Event()

    def run(self):
        smi = shutil.which("nvidia-smi")
        if not smi:
            return
        while not self.stop.is_set():
            try:
                out = subprocess.run([smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                      "-i", "0"], capture_output=True, text=True, timeout=10).stdout
                v = int(out.strip().splitlines()[0])
                self.peak = max(self.peak or 0, v)
            except Exception:
                pass
            self.stop.wait(1)


def build(cfg, wf, a, image_name):
    steps, mult = a.steps, cfg["size_multiple"]
    for nid, n in wf.items():
        t, i, title = n["class_type"], n["inputs"], n.get("_meta", {}).get("title", "").lower()
        if t == "CLIPTextEncode":
            i["text"] = a.negative if "negative" in title else a.prompt
        elif t in LATENT_NODES:
            i.update(width=a.width, height=a.height, length=a.frames)
        elif t == "KSampler":
            i.update(seed=a.seed, steps=steps)
            if a.cfg is not None:
                i["cfg"] = a.cfg
        elif t == "CreateVideo" and a.fps:
            i["fps"] = a.fps
        elif t == "SaveVideo":
            i["filename_prefix"] = "video/wan22"
        elif t == "LoadImage" and image_name:
            i["image"] = image_name

    adv = sorted((n for n in wf.values() if n["class_type"] == "KSamplerAdvanced"),
                 key=lambda n: n["inputs"]["start_at_step"])
    if adv:  # A14B: high-noise expert for first half, low-noise for the rest
        split = steps // 2
        for k, n in enumerate(adv):
            n["inputs"].update(noise_seed=a.seed, steps=steps)
            if a.cfg is not None:
                n["inputs"]["cfg"] = a.cfg
            if k == 0:
                n["inputs"]["end_at_step"] = split
            else:
                n["inputs"]["start_at_step"] = split

    has_loadimage = any(n["class_type"] == "LoadImage" for n in wf.values())
    if image_name and not has_loadimage:  # TI2V-5B: attach start image to the latent node
        nid = str(max(int(k) for k in wf) + 1)
        wf[nid] = {"class_type": "LoadImage", "inputs": {"image": image_name},
                   "_meta": {"title": "Start Image"}}
        for n in wf.values():
            if n["class_type"] == "Wan22ImageToVideoLatent":
                n["inputs"]["start_image"] = [nid, 0]
    if has_loadimage and not image_name:
        raise ComfyError("This workflow is image-to-video; pass --image.")
    return wf


def main():
    try:
        run_cli()
    except ComfyError as e:
        sys.exit(str(e))


def run_cli():
    cfg = load_config()
    d = cfg["defaults"]

    ap = argparse.ArgumentParser(description="Generate a Wan 2.2 video via ComfyUI")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=NEG_DEFAULT)
    ap.add_argument("--image", type=Path, help="start image -> image-to-video")
    ap.add_argument("--frames", type=int, default=d["frames"], help="4n+1, e.g. 49, 81, 121")
    ap.add_argument("--steps", type=int, default=d["steps"])
    ap.add_argument("--seed", type=int, default=None, help="random if omitted")
    ap.add_argument("--width", type=int, default=d["width"])
    ap.add_argument("--height", type=int, default=d["height"])
    ap.add_argument("--cfg", type=float, default=None, help=f"default {d['cfg']} (from workflow)")
    ap.add_argument("--fps", type=float, default=None, help=f"default {d['fps']}")
    ap.add_argument("--workflow", help=f"one of {list(cfg['workflows'])} or a path to an API JSON")
    ap.add_argument("--name", default="wan22", help="output name prefix in videos/")
    ap.add_argument("--server", default=cfg["server"])
    ap.add_argument("--outdir", type=Path, default=HERE / "videos")
    a = ap.parse_args()

    a.seed = a.seed if a.seed is not None else random.randint(0, 2**31 - 1)
    a.frames = fix_frames(a.frames)
    wf_key = a.workflow or ("i2v" if a.image else "t2v")
    if wf_key == "5b":
        cfg["size_multiple"] = 32
    a.width = fix_dim(a.width, cfg["size_multiple"], "width")
    a.height = fix_dim(a.height, cfg["size_multiple"], "height")

    wf_file, wf = load_workflow(cfg, wf_key)

    image_name = None
    if a.image:
        if not a.image.exists():
            raise ComfyError(f"Image not found: {a.image}")
        image_name = upload_image(a.server, a.image)
    wf = build(cfg, wf, a, image_name)

    print(f"Workflow {wf_file.name} | {a.width}x{a.height} | {a.frames} frames | "
          f"{a.steps} steps | seed {a.seed}" + (f" | image {a.image.name}" if a.image else ""))
    pid = queue(a.server, wf, uuid.uuid4().hex)
    print(f"Queued prompt {pid}")

    mon = VramMonitor()
    mon.start()
    t0 = time.time()
    last = 0
    while True:
        h = json.loads(http(a.server, f"/history/{pid}"))
        if pid in h:
            entry = h[pid]
            break
        if time.time() - last > 15:
            print(f"  ... {time.time() - t0:5.0f}s elapsed", flush=True)
            last = time.time()
        time.sleep(2)
    elapsed = time.time() - t0
    mon.stop.set()

    err = error_text(entry)
    if err:
        raise ComfyError(err)

    meta = {"prompt": a.prompt, "negative": a.negative, "width": a.width, "height": a.height,
            "frames": a.frames, "steps": a.steps, "seed": a.seed, "cfg": a.cfg, "fps": a.fps,
            "workflow": wf_key, "image": a.image.name if a.image else None,
            "seconds": round(elapsed), "peak_vram_gb": round(mon.peak / 1024, 1) if mon.peak else None}
    saved = save_outputs(a.server, entry, a.outdir, a.name, meta)
    if not saved:
        raise ComfyError(f"Finished in {elapsed:.0f}s but no video output was found: {entry.get('outputs')}")
    for dest in saved:
        print(f"Saved {dest}")
    print(f"Generation time: {elapsed:.0f}s" +
          (f" | peak VRAM: {mon.peak / 1024:.1f} GB" if mon.peak else ""))


if __name__ == "__main__":
    main()
