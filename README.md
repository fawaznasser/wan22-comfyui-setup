# Wan 2.2 local setup (ComfyUI)

This repo sets up the **Wan 2.2** video generation models locally with ComfyUI, in one command.
On the machine you run it on, the installer:

1. **Checks the hardware**: OS, Python, disk space, NVIDIA GPU, VRAM, driver and CUDA version. Then it picks a model tier:

   | VRAM | Tier | Models | Disk |
   |---|---|---|---|
   | < 12 GB | `gguf` | TI2V-5B GGUF (Q5_K_M if 8 GB or more, otherwise Q4_K_M) through ComfyUI-GGUF, with `--lowvram` | ~20 GB |
   | 12–23 GB | `5b` | TI2V-5B fp16 (text and image to video, 720p, 24 fps) | ~25 GB |
   | ≥ 24 GB | `a14b` | T2V-A14B + I2V-A14B fp8 (high-noise and low-noise experts), plus TI2V-5B | ~80 GB |

2. **Installs ComfyUI** into `C:\ai\` (Windows) or `~/ai/` (Linux), creates a venv, installs the PyTorch CUDA build that matches your driver, and checks that `torch.cuda.is_available()` returns True. It also adds ComfyUI-Manager, plus ComfyUI-GGUF on the `gguf` tier.
3. **Downloads the models** from `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` (and `QuantStack/Wan2.2-TI2V-5B-GGUF` for GGUF). It reads each repo's current file list, so renamed files are still found. Downloads resume after a failure, and file sizes are checked afterwards. It asks before downloading more than 50 GB.
4. **Writes the convenience files**: a `start_wan` launcher, `generate.py` (plus a `generate` wrapper), API-format workflows, and a README tailored to your GPU.
5. **Runs a test generation** if you want one. It makes a 49-frame, 832x480 clip at 20 steps through the ComfyUI HTTP API, then reports the time and peak VRAM.

## Requirements

- An NVIDIA GPU with a recent driver. The installer stops with suggestions if it doesn't find one.
- Python 3.10–3.13 (3.12 recommended) and git
- About 25 GB of free disk space, or about 80 GB for the A14B tier (models plus roughly 8 GB for PyTorch and the venv)

## Install on a new machine

**Windows**

```powershell
winget install Git.Git Python.Python.3.12      # if not installed yet
git clone https://github.com/fawaznasser/wan22-comfyui-setup.git
cd wan22-comfyui-setup
.\setup.bat                                    # or: powershell -ExecutionPolicy Bypass -File setup.ps1
```

**Linux** (including cloud GPU boxes such as RunPod or Vast.ai)

```bash
git clone https://github.com/fawaznasser/wan22-comfyui-setup.git
cd wan22-comfyui-setup
chmod +x setup.sh && ./setup.sh
```

### Installer options

```
--root PATH       install folder (default C:\ai or ~/ai)
--tier TIER       force gguf | 5b | a14b instead of auto-detecting
--quant Q         GGUF quant: Q4_K_M | Q5_K_M | Q6_K | Q8_0
--yes             no confirmation prompts
--test            run the test generation at the end
--no-test         skip the test generation
--skip-models     reuse already-downloaded models (re-write configs only)
```

You can safely re-run the installer. It skips models that are already downloaded and resumes partial downloads.

## Daily use

```bat
cd C:\ai
start_wan.bat                                         & rem terminal 1 - server at http://127.0.0.1:8188
generate.bat --prompt "A red fox trotting through snow, cinematic"   & rem terminal 2
generate.bat --prompt "She turns and smiles" --image photo.jpg       & rem image-to-video
```

On Linux, use `./start_wan.sh` and `./generate.sh` instead.
Videos are saved to `videos/` with a timestamp in the name.

`generate.py` takes these options: `--prompt --negative --image --frames --steps --seed --width --height --cfg --fps --workflow --name`

### Frame and resolution rules

- The frame count must be **4n + 1** (17, 33, 49, 81, 121 ...). For TI2V-5B at 24 fps, 121 frames is about 5 s. For A14B at 16 fps, 81 frames is about 5 s.
- Width and height must be multiples of 32 for TI2V-5B and multiples of 16 for A14B. `generate.py` rounds them for you.
- If you run out of memory, step down in this order: lower resolution, then fewer frames, then a smaller tier (`--tier gguf`), then `--lowvram`.

## Repo layout

```
setup.bat / setup.ps1 / setup.sh   entry points (find Python, run install.py)
install.py                          main installer
scripts/download_models.py          Hugging Face downloader (runs in the ComfyUI venv)
scripts/generate.py                 API client, copied to the install folder
workflows/*.json                    API-format workflows (5B, T2V-A14B, I2V-A14B)
docs/AGENT_PROMPT.md                same setup as a prompt for a coding agent
```

The workflows mirror ComfyUI's built-in Wan 2.2 templates. If a ComfyUI update renames a node, open
**Workflow → Browse Templates → Video**, load the Wan 2.2 template, and use **Export (API)** to save it over the
matching file in `C:\ai\workflows\`. Keep the node titles "Positive Prompt" and "Negative Prompt".
