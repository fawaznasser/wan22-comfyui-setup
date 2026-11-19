#!/usr/bin/env python3
"""
Wan 2.2 + ComfyUI installer.

Detects the NVIDIA GPU on THIS machine, picks the right Wan 2.2 model tier,
installs ComfyUI into a venv with the matching CUDA build of PyTorch,
downloads the models, writes workflows / launcher / generate.py / README,
and optionally runs a test generation.

Usage (normally via setup.ps1 / setup.bat / setup.sh):
    python install.py                 # interactive
    python install.py --yes --test    # unattended + test render
    python install.py --tier 5b       # force a tier (gguf | 5b | a14b)
"""
import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IS_WIN = os.name == "nt"

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"
MANAGER_REPO = "https://github.com/ltdrdata/ComfyUI-Manager"
GGUF_NODE_REPO = "https://github.com/city96/ComfyUI-GGUF"

TIERS = {
    "gguf": "Wan 2.2 TI2V-5B GGUF ({quant}) via ComfyUI-GGUF, started with --lowvram",
    "5b": "Wan 2.2 TI2V-5B fp16 (text-to-video + image-to-video, 720p, 24 fps)",
    "a14b": "Wan 2.2 T2V-A14B + I2V-A14B fp8 (high-noise + low-noise experts), plus TI2V-5B",
}

# Per-tier generation defaults written to wan_config.json
DEFAULTS = {
    "gguf": {"width": 832, "height": 480, "frames": 49, "steps": 20, "fps": 24, "cfg": 5.0},
    "5b": {"width": 1280, "height": 704, "frames": 121, "steps": 30, "fps": 24, "cfg": 5.0},
    "a14b": {"width": 1280, "height": 720, "frames": 81, "steps": 20, "fps": 16, "cfg": 3.5},
}
TEST = {"width": 832, "height": 480, "frames": 49, "steps": 20}
TEST_PROMPT = ("A red fox trotting through a snowy pine forest at sunrise, slow tracking shot, "
               "cinematic lighting, shallow depth of field")
TEST_NEGATIVE = "blurry, distorted, extra limbs, static, low quality, watermark, text"


# --------------------------------------------------------------------------- helpers
def log(msg):
    print(f"\n==> {msg}", flush=True)


def run(cmd, **kw):
    cmd = [str(c) for c in cmd]
    print("   $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def ask(question, assume_yes):
    if assume_yes:
        print(f"{question} [auto-yes]")
        return True
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def free_gb(path: Path):
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1024**3


def dir_size_gb(path: Path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1024**3


def venv_python(comfy: Path):
    return comfy / "venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")


# --------------------------------------------------------------------------- step 1
def detect_gpu():
    smi = shutil.which("nvidia-smi")
    if not smi and IS_WIN and Path(r"C:\Windows\System32\nvidia-smi.exe").exists():
        smi = r"C:\Windows\System32\nvidia-smi.exe"
    if not smi:
        return None
    try:
        q = subprocess.run(
            [smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout.strip().splitlines()
    except (subprocess.CalledProcessError, OSError):
        return None
    if not q:
        return None
    name, mem, drv = [s.strip() for s in q[0].split(",")]
    full = subprocess.run([smi], capture_output=True, text=True).stdout
    m = re.search(r"CUDA Version:\s*([\d.]+)", full)
    return {"name": name, "vram_gb": int(float(mem)) / 1024, "driver": drv,
            "cuda": m.group(1) if m else None, "count": len(q)}


def choose_tier(vram_gb):
    # thresholds slightly under the nominal size: a "12 GB" card reports ~11.99 GiB
    if vram_gb < 11.5:
        return "gguf"
    if vram_gb < 23.5:
        return "5b"
    return "a14b"


def torch_index(cuda):
    if not cuda:
        sys.exit("Could not read the CUDA version from nvidia-smi. Update your NVIDIA driver.")
    v = tuple(int(x) for x in cuda.split(".")[:2])
    for need, tag in [((12, 8), "cu128"), ((12, 6), "cu126"), ((12, 4), "cu124"),
                      ((12, 1), "cu121"), ((11, 8), "cu118")]:
        if v >= need:
            return tag
    sys.exit(f"Driver supports only CUDA {cuda}; PyTorch needs >= 11.8. Update your NVIDIA driver.")


NO_GPU_HELP = """
No NVIDIA GPU detected (nvidia-smi not found or failed). Wan 2.2 needs CUDA to run at
a usable speed. Options:
  1. Install/update the NVIDIA driver if this machine does have an NVIDIA card, then re-run.
  2. Run this installer on a machine with an NVIDIA GPU (8 GB+ VRAM; 24 GB for A14B).
  3. Rent a cloud GPU (RunPod / Vast.ai, e.g. RTX 4090 24 GB) and run setup.sh there.
  4. Use a free Kaggle / Colab GPU notebook.
"""


# --------------------------------------------------------------------------- step 2
def install_comfy(root: Path, tier: str, cuda: str):
    comfy = root / "ComfyUI"
    root.mkdir(parents=True, exist_ok=True)
    if not (comfy / ".git").exists():
        run(["git", "clone", "--depth", "1", COMFY_REPO, comfy])
    else:
        print("   ComfyUI already present - trying fast-forward update")
        subprocess.run(["git", "-C", str(comfy), "pull", "--ff-only"])

    py = venv_python(comfy)
    if not py.exists():
        run([sys.executable, "-m", "venv", comfy / "venv"])
    run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel"])

    tag = torch_index(cuda)
    log(f"Installing PyTorch ({tag}) for CUDA {cuda}")
    run([py, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
         "--index-url", f"https://download.pytorch.org/whl/{tag}"])
    run([py, "-m", "pip", "install", "-r", comfy / "requirements.txt"])
    run([py, "-m", "pip", "install", "-U", "huggingface_hub"])

    log("Verifying CUDA from inside the venv")
    out = subprocess.run([str(py), "-c",
                          "import torch; print(torch.__version__, torch.cuda.is_available(), "
                          "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"],
                         capture_output=True, text=True)
    print("   " + (out.stdout.strip() or out.stderr.strip()))
    if " True " not in f" {out.stdout} ":
        sys.exit("torch.cuda.is_available() is False. Check the driver / CUDA build above.")

    nodes = comfy / "custom_nodes"
    repos = [MANAGER_REPO] + ([GGUF_NODE_REPO] if tier == "gguf" else [])
    for repo in repos:
        dest = nodes / repo.rstrip("/").split("/")[-1]
        if not dest.exists():
            run(["git", "clone", "--depth", "1", repo, dest])
        req = dest / "requirements.txt"
        if req.exists():
            run([py, "-m", "pip", "install", "-r", req])
    return comfy


# --------------------------------------------------------------------------- step 3
def download_models(comfy: Path, tier: str, quant: str, assume_yes: bool):
    manifest = comfy.parent / "models_manifest.json"
    cmd = [venv_python(comfy), "-I", HERE / "scripts" / "download_models.py",
           "--comfy", comfy, "--tier", tier, "--quant", quant, "--manifest", manifest]
    if assume_yes:
        cmd.append("--yes")
    run(cmd)
    return json.loads(manifest.read_text())


# --------------------------------------------------------------------------- workflows
def write_workflows(root: Path, tier: str, m: dict):
    src, out = HERE / "workflows", root / "workflows"
    out.mkdir(parents=True, exist_ok=True)

    def patch(name, mapping, dest=None, gguf=False):
        wf = json.loads((src / name).read_text(encoding="utf-8"))
        for node in wf.values():
            title = node.get("_meta", {}).get("title", "")
            if title not in mapping:
                continue
            if gguf and node["class_type"] == "UNETLoader":
                node["class_type"] = "UnetLoaderGGUF"
                node["inputs"] = {"unet_name": mapping[title]}
                continue
            key = next(k for k in node["inputs"] if k.endswith("_name"))
            node["inputs"][key] = mapping[title]
        (out / (dest or name)).write_text(json.dumps(wf, indent=2), encoding="utf-8")
        return dest or name

    te = m["text_encoder"]
    if tier == "gguf":
        f = patch("ti2v_5b.json", {"Load Diffusion Model": m["gguf"], "Load Text Encoder": te,
                                   "Load VAE": m["vae22"]}, "ti2v_5b_gguf.json", gguf=True)
        return {"t2v": f, "i2v": f}
    f5 = patch("ti2v_5b.json", {"Load Diffusion Model": m["diffusion_5b"],
                                "Load Text Encoder": te, "Load VAE": m["vae22"]})
    if tier == "5b":
        return {"t2v": f5, "i2v": f5}
    t2v = patch("t2v_a14b.json", {"Load High Noise Model": m["t2v_high"],
                                  "Load Low Noise Model": m["t2v_low"],
                                  "Load Text Encoder": te, "Load VAE": m["vae21"]})
    i2v = patch("i2v_a14b.json", {"Load High Noise Model": m["i2v_high"],
                                  "Load Low Noise Model": m["i2v_low"],
                                  "Load Text Encoder": te, "Load VAE": m["vae21"]})
    return {"t2v": t2v, "i2v": i2v, "5b": f5}


# --------------------------------------------------------------------------- step 5
def write_convenience(root: Path, comfy: Path, tier: str, gpu: dict, workflows: dict, quant: str):
    flags = ["--listen", "127.0.0.1", "--port", "8188"] + (["--lowvram"] if tier == "gguf" else [])
    cfg = {"tier": tier, "quant": quant if tier == "gguf" else None, "gpu": gpu["name"],
           "vram_gb": round(gpu["vram_gb"], 1), "comfy_dir": str(comfy),
           "server": "http://127.0.0.1:8188", "launch_flags": flags,
           "size_multiple": 16 if tier == "a14b" else 32,
           "workflows": workflows, "defaults": DEFAULTS[tier]}
    (root / "wan_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for f in ("generate.py", "webui.py", "webui.html"):
        shutil.copy2(HERE / "scripts" / f, root / f)
    (root / "videos").mkdir(exist_ok=True)
    venv_py = venv_python(comfy)
    if IS_WIN:
        (root / "start_ui.bat").write_text(
            "@echo off\r\nrem Starts ComfyUI (if not running) + Wan Studio, opens the browser.\r\n"
            f'"{venv_py}" "%~dp0webui.py" %*\r\n', encoding="utf-8")
    else:
        ui = root / "start_ui.sh"
        ui.write_text("#!/usr/bin/env bash\n# Starts ComfyUI (if not running) + Wan Studio.\n"
                      f'exec "{venv_py}" "$(dirname "$0")/webui.py" "$@"\n', encoding="utf-8")
        ui.chmod(0o755)

    if IS_WIN:
        launcher = root / "start_wan.bat"
        launcher.write_text(
            "@echo off\r\n"
            'cd /d "%~dp0ComfyUI"\r\n'
            "call venv\\Scripts\\activate.bat\r\n"
            f"python main.py {' '.join(flags)} %*\r\n", encoding="utf-8")
        gen = root / "generate.bat"
        gen.write_text('@echo off\r\n"%~dp0ComfyUI\\venv\\Scripts\\python.exe" "%~dp0generate.py" %*\r\n',
                       encoding="utf-8")
    else:
        launcher = root / "start_wan.sh"
        launcher.write_text(
            "#!/usr/bin/env bash\n"
            'cd "$(dirname "$0")/ComfyUI"\n'
            "source venv/bin/activate\n"
            f'exec python main.py {" ".join(flags)} "$@"\n', encoding="utf-8")
        gen = root / "generate.sh"
        gen.write_text('#!/usr/bin/env bash\nD="$(dirname "$0")"\n'
                       'exec "$D/ComfyUI/venv/bin/python" "$D/generate.py" "$@"\n', encoding="utf-8")
        for f in (launcher, gen):
            f.chmod(0o755)

    write_readme(root, tier, gpu, quant, launcher.name, gen.name)
    return launcher, gen


def write_readme(root, tier, gpu, quant, launcher, gen):
    d = DEFAULTS[tier]
    run_gen = gen if IS_WIN else f"./{gen}"
    run_launch = launcher if IS_WIN else f"./{launcher}"
    rec = {
        "gguf": f"""- Model: TI2V-5B GGUF {quant}, ComfyUI runs with `--lowvram`.
- Start at 832x480, 49 frames, 20 steps. Go up to 1280x704 only if it fits.
- On OOM, lower the resolution first, then the frames.""",
        "5b": """- Model: TI2V-5B fp16.
- Native: 1280x704 (or 704x1280), 121 frames, 30 steps, cfg 5, 24 fps.
- Width and height must be multiples of 32. For quick drafts use 832x480 / 49 frames / 20 steps.""",
        "a14b": """- Models: T2V-A14B + I2V-A14B fp8 (two experts: high-noise for the first half of the steps, low-noise for the rest), plus TI2V-5B.
- A14B native: 1280x720 or 832x480, 81 frames, 20 steps, cfg 3.5, **16 fps**.
- Use `--workflow 5b` for the faster TI2V-5B model (24 fps, 1280x704).""",
    }[tier]
    ui = "start_ui.bat" if IS_WIN else "./start_ui.sh"
    text = f"""# Wan 2.2 (local, ComfyUI)

Installed for **{gpu['name']}** ({gpu['vram_gb']:.1f} GB VRAM). Tier: **{tier}**.

## Easiest: Wan Studio (browser UI)

```
{ui}
```

This starts ComfyUI and opens **http://127.0.0.1:7860** in your browser. There you can type a prompt, drop in an image if you want image-to-video, pick an aspect ratio, quality and length, then click **Generate**. You get a live progress bar, the video plays in the page, and every result is kept in a gallery. Press Ctrl+C in that window to stop everything.

## Command line: start the server

```
{run_launch}
```

The UI is at http://127.0.0.1:8188. Leave this window open while you generate.

## Generate (in a second terminal)

```
{run_gen} --prompt "A red fox trotting through snow, cinematic" --frames {d['frames']} --steps {d['steps']}
{run_gen} --prompt "The woman turns and smiles" --image photo.jpg        # image-to-video
{run_gen} --help
```

Each result is copied to `videos/` with a timestamped name. The raw output also stays in `ComfyUI/output/`.

Options: `--prompt --negative --image --frames --steps --seed --width --height --cfg --fps --workflow`

## Recommended settings for your GPU

{rec}

Defaults are in `wan_config.json` (currently {d['width']}x{d['height']}, {d['frames']} frames, {d['steps']} steps).

## Frame-count rules

- The frame count must be **4n + 1**: 17, 33, 49, 65, 81, 97, 121, ...
- At 24 fps (5B): 49 frames is about 2 s and 121 frames is about 5 s.
- At 16 fps (A14B): 81 frames is about 5 s.
- `generate.py` rounds invalid values to the nearest valid count.

## Running out of memory (OOM)

Step down in this order: lower resolution, then fewer frames, then a smaller model (`--workflow 5b`, or re-run the installer with `--tier gguf`), then add `--lowvram` to the launcher.

## Files

- `ComfyUI/`: the app, its venv, and the models under `ComfyUI/models/`
- `workflows/`: API-format workflows used by `generate.py`. You can also drag them into the UI.
- `wan_config.json`: tier, model files and defaults
"""
    (root / "README.md").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- step 4
def port_open(port=8188):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def run_test(root: Path, comfy: Path, flags):
    log("4. Launching ComfyUI for a test generation")
    if port_open():
        print("   Something is already listening on 8188 - using it.")
        server = None
    else:
        logf = open(root / "comfyui_test.log", "w", encoding="utf-8")
        server = subprocess.Popen([str(venv_python(comfy)), "main.py", *flags],
                                  cwd=comfy, stdout=logf, stderr=subprocess.STDOUT)
        for _ in range(300):
            if port_open():
                break
            if server.poll() is not None:
                sys.exit(f"ComfyUI exited early - see {root / 'comfyui_test.log'}")
            time.sleep(1)
        else:
            server.terminate()
            sys.exit("ComfyUI did not start within 5 minutes.")
        print("   ComfyUI is serving on http://127.0.0.1:8188")
    try:
        cmd = [venv_python(comfy), root / "generate.py", "--prompt", TEST_PROMPT,
               "--negative", TEST_NEGATIVE, "--frames", TEST["frames"], "--steps", TEST["steps"],
               "--width", TEST["width"], "--height", TEST["height"], "--seed", 42,
               "--name", "test_fox"]
        r = subprocess.run([str(c) for c in cmd])
        ok = r.returncode == 0
    finally:
        if server:
            server.terminate()
            try:
                server.wait(30)
            except subprocess.TimeoutExpired:
                server.kill()
    if not ok:
        print(f"   Test failed - see output above and {root / 'comfyui_test.log'}")
    return ok


# --------------------------------------------------------------------------- main
def main():
    default_root = Path(r"C:\ai") if IS_WIN else Path.home() / "ai"
    ap = argparse.ArgumentParser(description="Install Wan 2.2 + ComfyUI")
    ap.add_argument("--root", type=Path, default=default_root, help=f"install folder (default {default_root})")
    ap.add_argument("--tier", choices=TIERS, help="override automatic tier choice")
    ap.add_argument("--quant", choices=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"], help="GGUF quant (gguf tier)")
    ap.add_argument("--yes", "-y", action="store_true", help="don't ask for confirmation")
    ap.add_argument("--test", action="store_true", help="run the test generation without asking")
    ap.add_argument("--no-test", action="store_true", help="skip the test generation")
    ap.add_argument("--skip-models", action="store_true", help="don't download models (reuse manifest)")
    args = ap.parse_args()
    root = args.root.expanduser().resolve()

    if not ((3, 10) <= sys.version_info[:2] <= (3, 13)):
        sys.exit(f"Python {platform.python_version()} found; use Python 3.10-3.13 (3.12 recommended).")
    if not shutil.which("git"):
        sys.exit("git is required (Windows: winget install Git.Git, Ubuntu: sudo apt install git).")

    log("1. Detecting hardware")
    print(f"   OS:        {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"   Python:    {platform.python_version()} ({sys.executable})")
    print(f"   Disk free: {free_gb(root):.1f} GB at {root}")
    gpu = detect_gpu()
    if not gpu:
        print(NO_GPU_HELP)
        sys.exit(1)
    print(f"   GPU:       {gpu['name']}  ({gpu['vram_gb']:.1f} GB VRAM)"
          + (f"  [+{gpu['count'] - 1} more, using GPU 0]" if gpu["count"] > 1 else ""))
    print(f"   Driver:    {gpu['driver']}   CUDA (driver max): {gpu['cuda']}")

    tier = args.tier or choose_tier(gpu["vram_gb"])
    quant = args.quant or ("Q5_K_M" if gpu["vram_gb"] >= 8 else "Q4_K_M")
    reason = ("forced with --tier" if args.tier else
              {"gguf": "VRAM < 12 GB, so use a quantized 5B model with --lowvram",
               "5b": "12-23 GB VRAM fits TI2V-5B fp16 at 720p",
               "a14b": ">= 24 GB VRAM fits the A14B fp8 experts"}[tier])
    print(f"\n   Chosen: {TIERS[tier].format(quant=quant)}\n   Why:    {reason}")
    print(f"   Approx. disk needed: {dict(gguf='~20', a14b='~80').get(tier, '~25')} GB (models + ~8 GB PyTorch/venv)")
    if not ask("Proceed with this setup?", args.yes):
        sys.exit("Aborted.")

    log(f"2. Installing ComfyUI into {root}")
    comfy = install_comfy(root, tier, gpu["cuda"])

    log("3. Downloading models")
    manifest = root / "models_manifest.json"
    models = json.loads(manifest.read_text()) if args.skip_models and manifest.exists() \
        else download_models(comfy, tier, quant, args.yes)

    log("5. Writing workflows, launcher, generate.py, README")
    workflows = write_workflows(root, tier, models)
    launcher, gen = write_convenience(root, comfy, tier, gpu, workflows, quant)
    flags = json.loads((root / "wan_config.json").read_text())["launch_flags"]

    tested = None
    if not args.no_test and (args.test or ask("Run a short test generation now (a few minutes)?", False)):
        tested = run_test(root, comfy, flags)

    log("Summary")
    print(f"   Installed to:   {root}")
    print(f"   ComfyUI:        {comfy}")
    print(f"   Models:         {comfy / 'models'}")
    print(f"   Workflows:      {root / 'workflows'}")
    print(f"   Disk used:      {dir_size_gb(root):.1f} GB")
    print(f"   Test result:    {'passed' if tested else 'FAILED' if tested is False else 'skipped'}")
    pre = "" if IS_WIN else "./"
    print("\n   Day to day:")
    print(f"     cd {root}")
    print(f"     {pre}start_ui{'.bat' if IS_WIN else '.sh'}                           # browser UI at http://127.0.0.1:7860")
    print("   or the command line:")
    print(f"     {pre}{launcher.name}                          # terminal 1: start server")
    print(f'     {pre}{gen.name} --prompt "your prompt"       # terminal 2: make a video')
    print(f"   Read {root / 'README.md'} for settings.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"\nCommand failed (exit {e.returncode}): {' '.join(map(str, e.cmd))}")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Re-run the same command to resume - finished steps are skipped.")
