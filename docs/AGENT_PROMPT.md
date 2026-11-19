# Agent prompt (alternative to the scripts)

If you'd rather have a coding agent such as Claude Code do the setup step by step on the target machine, paste the prompt
below into it. The scripts in this repo do the same thing without an agent.

```text
Set up Wan 2.2 video generation locally on this machine using ComfyUI. Work step by step, verify each step before moving on, and stop to ask me before anything destructive or before downloads larger than 50 GB.

1. DETECT HARDWARE
   - Report OS, Python version, free disk space, NVIDIA GPU model, VRAM, driver and CUDA version (nvidia-smi). If there is no NVIDIA GPU, stop and tell me my options.
   - Choose the model based on VRAM:
     • < 12 GB  → Wan 2.2 TI2V-5B as GGUF (Q4_K_M or Q5_K_M) via the ComfyUI-GGUF custom node
     • 12–23 GB → Wan 2.2 TI2V-5B fp16 (text-to-video and image-to-video, 720p, 24fps)
     • ≥ 24 GB  → Wan 2.2 T2V-A14B and I2V-A14B fp8 (high-noise + low-noise expert models), plus TI2V-5B
   - Tell me which option you picked and why before downloading.

2. INSTALL COMFYUI
   - Create a folder ~/ai/ (or C:\ai\ on Windows). Clone https://github.com/comfyanonymous/ComfyUI into it.
   - Create a Python venv inside ComfyUI, install the PyTorch build that matches my CUDA version (from pytorch.org instructions), then `pip install -r requirements.txt`.
   - Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` must print True.
   - Install ComfyUI-Manager into custom_nodes (and ComfyUI-GGUF if using GGUF).

3. DOWNLOAD MODELS (use huggingface-cli / hf download, resume on failure)
   From Comfy-Org/Wan_2.2_ComfyUI_Repackaged (and the matching GGUF repo if needed), place:
   - Diffusion model(s)  → ComfyUI/models/diffusion_models/   (e.g. wan2.2_ti2v_5B_fp16.safetensors, or the A14B high/low noise fp8 files)
   - Text encoder        → ComfyUI/models/text_encoders/      (umt5_xxl_fp8_e4m3fn_scaled.safetensors)
   - VAE                 → ComfyUI/models/vae/                (wan2.2_vae.safetensors for the 5B; wan_2.1_vae.safetensors for the A14B models)
   - GGUF files go in ComfyUI/models/unet/ if using the GGUF loader.
   Check the repo's current file list first — exact filenames may have changed — and confirm sizes after download.

4. LAUNCH AND TEST
   - Start ComfyUI (`python main.py`, add `--lowvram` if VRAM < 12 GB) and confirm it serves on http://127.0.0.1:8188.
   - Use the built-in Wan 2.2 template (Workflow → Browse Templates → Video) as the base. Save a copy of the API-format workflow JSON to ~/ai/workflows/.
   - Run one short test generation through the ComfyUI HTTP API (POST /prompt) with:
     prompt: "A red fox trotting through a snowy pine forest at sunrise, slow tracking shot, cinematic lighting, shallow depth of field"
     negative: "blurry, distorted, extra limbs, static, low quality, watermark, text"
     low settings for the test: ~480p–704p, 49 frames, 20 steps
   - Confirm an MP4 appears in ComfyUI/output/ and report generation time and peak VRAM.

5. CONVENIENCE
   - Create a launcher script (start_wan.sh or start_wan.bat) that activates the venv and starts ComfyUI with the right flags.
   - Create a small Python script `generate.py` that takes --prompt, --negative, --image (optional, for image-to-video), --frames, --steps, --seed, --width, --height, sends the saved workflow to the ComfyUI API, waits for completion, and copies the result to ~/ai/videos/ with a timestamped name.
   - Write a README.md in ~/ai/ explaining how to start it, how to generate, recommended settings for my GPU, and frame-count rules (frames = 4n+1; 121 frames ≈ 5s at 24fps).

6. TROUBLESHOOTING
   If anything fails (CUDA mismatch, OOM, missing nodes, wrong filenames), diagnose the actual error, fix it, and tell me what you changed. On OOM, step down: lower resolution → fewer frames → fp8/GGUF → --lowvram.

Finish with a summary: what was installed where, total disk used, test result, and the exact commands I run day to day.
```
