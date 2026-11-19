"""
Download Wan 2.2 model files from Hugging Face into ComfyUI/models/.

Run with the ComfyUI venv python (needs huggingface_hub). It reads the repos'
current file lists, so a renamed file is matched by pattern rather than a
hard-coded path. It skips files already present at the right size and
resumes partial downloads.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# Comfy-Org renamed Wan_2.2_ComfyUI_Repackaged -> Wan-2.2; keep the old names as fallbacks.
R22 = ["Comfy-Org/Wan-2.2", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"]
R21 = ["Comfy-Org/Wan_2.1_ComfyUI_repackaged", "Comfy-Org/Wan-2.1"]
GGUF_REPOS = ["QuantStack/Wan2.2-TI2V-5B-GGUF"]

# key -> (models subfolder, repos to search, regex patterns: strict first, then looser)
FILES = {
    "diffusion_5b": ("diffusion_models", R22, [r"wan2\.2_ti2v_5B_fp16\.safetensors$",
                                                r"ti2v.*5b.*fp16.*\.safetensors$"]),
    "vae22": ("vae", R22, [r"wan2\.2_vae\.safetensors$"]),
    "vae21": ("vae", R22 + R21, [r"wan_2\.1_vae\.safetensors$", r"wan.?2\.1.*vae.*\.safetensors$"]),
    "text_encoder": ("text_encoders", R22 + R21, [r"umt5_xxl_fp8_e4m3fn_scaled\.safetensors$",
                                                  r"umt5.*xxl.*fp8.*\.safetensors$"]),
    "t2v_high": ("diffusion_models", R22, [r"wan2\.2_t2v_high_noise_14B_fp8_scaled\.safetensors$",
                                            r"t2v.*high.*14b.*fp8.*\.safetensors$"]),
    "t2v_low": ("diffusion_models", R22, [r"wan2\.2_t2v_low_noise_14B_fp8_scaled\.safetensors$",
                                           r"t2v.*low.*14b.*fp8.*\.safetensors$"]),
    "i2v_high": ("diffusion_models", R22, [r"wan2\.2_i2v_high_noise_14B_fp8_scaled\.safetensors$",
                                            r"i2v.*high.*14b.*fp8.*\.safetensors$"]),
    "i2v_low": ("diffusion_models", R22, [r"wan2\.2_i2v_low_noise_14B_fp8_scaled\.safetensors$",
                                           r"i2v.*low.*14b.*fp8.*\.safetensors$"]),
}

TIER_KEYS = {
    "gguf": ["gguf", "vae22", "text_encoder"],
    "5b": ["diffusion_5b", "vae22", "text_encoder"],
    "a14b": ["t2v_high", "t2v_low", "i2v_high", "i2v_low", "vae21",
             "diffusion_5b", "vae22", "text_encoder"],
}

_trees = {}


def tree(api, repo, sub):
    """Files under split_files/<sub> (Comfy layout) or the whole repo; [] if the repo is unavailable."""
    key = (repo, sub)
    if key not in _trees:
        files = []
        for path in ([f"split_files/{sub}"] if sub != "unet" else []) + [None]:
            try:
                files = [f for f in api.list_repo_tree(repo, path_in_repo=path, recursive=True)
                         if getattr(f, "size", None) is not None]
            except Exception:
                files = []
            if files:
                break
        _trees[key] = files
    return _trees[key]


def resolve(api, sub, repos, patterns):
    for pat in patterns:
        for repo in repos:
            hits = [f for f in tree(api, repo, sub) if re.search(pat, f.path, re.I)]
            if hits:
                f = min(hits, key=lambda f: len(f.path))
                return repo, f.path, f.size
    return None


def fetch(repo, path, size, dest_dir: Path, staging: Path):
    dest = dest_dir / Path(path).name
    if dest.exists() and dest.stat().st_size == size:
        print(f"   [skip] {dest.name} already present ({size / 1024**3:.2f} GB)")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"   [get ] {repo}/{path} ({size / 1024**3:.2f} GB)", flush=True)
    for attempt in range(1, 6):
        try:
            p = hf_hub_download(repo, path, local_dir=staging)
            break
        except Exception as e:  # network hiccups: retry (download resumes)
            wait = 15 * attempt
            print(f"   download error ({e.__class__.__name__}: {e}); retry {attempt}/5 in {wait}s")
            time.sleep(wait)
    else:
        sys.exit(f"Failed to download {path} after 5 attempts. Re-run to resume.")
    os.replace(p, dest)
    got = dest.stat().st_size
    if got != size:
        sys.exit(f"Size mismatch for {dest.name}: expected {size}, got {got}. Delete it and re-run.")
    print(f"   [ ok ] {dest} ({got / 1024**3:.2f} GB)")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", type=Path, required=True)
    ap.add_argument("--tier", choices=TIER_KEYS, required=True)
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    api = HfApi()
    models = a.comfy / "models"
    plan = []
    print("   Checking current repo file lists on Hugging Face ...")
    for key in TIER_KEYS[a.tier]:
        if key == "gguf":
            q = re.escape(a.quant)
            spec = ("unet", GGUF_REPOS, [rf"ti2v.*5b.*{q}\.gguf$", rf"{q}\.gguf$"])
        else:
            spec = FILES[key]
        sub, repos, pats = spec
        hit = resolve(api, sub, repos, pats)
        if not hit:
            sys.exit(f"Could not find a file for '{key}' in {repos} (patterns {pats}). "
                     "The repo layout may have changed - check it on huggingface.co.")
        plan.append((key, sub, *hit))

    todo = 0
    for key, sub, repo, path, size in plan:
        exists = (models / sub / Path(path).name)
        have = exists.exists() and exists.stat().st_size == size
        todo += 0 if have else size
        print(f"   {key:13s} -> models/{sub}/{Path(path).name:55s} {size / 1024**3:6.2f} GB"
              f"{'  (present)' if have else ''}")
    todo_gb = todo / 1024**3
    free = shutil.disk_usage(models if models.exists() else a.comfy).free / 1024**3
    print(f"   To download: {todo_gb:.1f} GB   Free: {free:.1f} GB")
    if todo_gb > free - 5:
        sys.exit("Not enough free disk space (keeping a 5 GB margin).")
    if todo_gb > 50 and not a.yes:
        if input(f"   Download {todo_gb:.1f} GB? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("Aborted before download.")

    staging = models / ".hf_staging"
    result = {}
    for key, sub, repo, path, size in plan:
        result[key] = fetch(repo, path, size, models / sub, staging).name
    shutil.rmtree(staging, ignore_errors=True)
    a.manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"   Manifest written: {a.manifest}")


if __name__ == "__main__":
    main()
