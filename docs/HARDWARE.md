# Hardware Requirements and Stable Diffusion Backend Setup

Soupy itself is light. The heavy work — chat inference, embeddings, image generation — happens in services Soupy talks to. This document covers what those services actually need, and how to stand up the bundled Stable Diffusion backend if you want image generation.

## Hardware Requirements

- 16GB+ system RAM on the bot host (more if you also run LM Studio there).
- 24GB GPU on the SD backend host (defaults are tuned for SD 3.5 Medium with `qfloat8` quantization; smaller VRAM may work for SD 1.5 / smaller SDXL checkpoints — see below).
- LLM and SD backends can each run on a separate machine on the LAN; the bot itself is light.

## Stable Diffusion Backend Setup

**Skip this entire section if you don't want image generation.** The bot runs fine without `/sd`, `/img2img`, `/inpaint`, and `/outpaint` — those commands will simply error out, and everything else (chat, search, daily posts, Bluesky, etc.) keeps working.

If you do want image generation, Soupy's `sd-api/` directory ships a reference FastAPI backend that the bot talks to over HTTP. It is **a separate process with its own dependencies and its own Python environment**, typically run on a different machine on the LAN — the GPU host — while the bot itself runs somewhere lighter.

### What's in `sd-api/`

- `sd-api/sd_api.py` — Main FastAPI backend (Linux / NVIDIA CUDA).
- `sd-api/sd_api-mac.py` — Apple Silicon variant — same API, uses MPS (Metal) instead of CUDA, falls back to CPU.
- `sd-api/requirements.txt` — Dependency list for the SD host (Linux/CUDA).
- `sd-api/requirements-m1-mac.txt` — Dependency list for Apple Silicon.
- `sd-api/M1_MAC_SETUP.md` — Mac-specific setup notes — read this if you're running the backend on Apple Silicon.

### Install (Linux / NVIDIA)

The SD host needs its own venv. **Do not** install these into the bot's venv — the bot's `requirements.txt` deliberately excludes the heavy SD-only stack (`diffusers`, `optimum-quanto`, `rembg`, `transformers`, etc.).

```bash
# On the GPU host
git clone https://github.com/sneezeparty/soupy.git    # only sd-api/ is strictly needed
cd soupy
python -m venv .venv-sd
source .venv-sd/bin/activate

# PyTorch with CUDA 11.8 — install before sd-api/requirements.txt
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

pip install -r sd-api/requirements.txt

python sd-api/sd_api.py
```

The first launch downloads the configured model from Hugging Face (default: `stabilityai/stable-diffusion-3.5-medium`) into the standard HF cache (`~/.cache/huggingface/`). This can be tens of GB and take a while depending on your connection. Subsequent launches use the cache.

### Install (Apple Silicon)

```bash
# On the Mac
git clone https://github.com/sneezeparty/soupy.git
cd soupy
python -m venv .venv-sd
source .venv-sd/bin/activate

# Standard (non-CUDA) PyTorch — MPS support is built in
pip install -r sd-api/requirements-m1-mac.txt

python sd-api/sd_api-mac.py
```

`sd_api-mac.py` auto-detects MPS at startup. See [`sd-api/M1_MAC_SETUP.md`](../sd-api/M1_MAC_SETUP.md) for verification steps, performance notes, and troubleshooting.

### Ports and endpoints

The backend is **a single FastAPI app listening on port `8000`**. All endpoints — `/sd`, `/sd_img2img`, `/sd_inpaint`, `/outpaint_hybrid`, `/remove_background`, `/upscale`, `/health` — are served from that one process. The `host="0.0.0.0", port=8000` binding is currently hardcoded in `sd_api.py` / `sd_api-mac.py`; change them in code if you need a different port.

> **Note on the `:8001` defaults in `.env-stable.example`.** The example file ships with `SD_IMG2IMG_URL` and `SD_INPAINT_URL` pointing at `:8001`, which would be correct for a multi-process setup but does **not** match the bundled single-process backend. The bot has fallback logic that derives both URLs from `SD_SERVER_URL` if the dedicated values are unreachable, so things still work — but the cleanest config is to point all four bot-side env vars at the same `:8000` host.

### Bot-side configuration

In `.env-stable`, point all the SD endpoints at the backend host:

```bash
SD_SERVER_URL=http://<sd-host>:8000/
SD_IMG2IMG_URL=http://<sd-host>:8000/sd_img2img
SD_INPAINT_URL=http://<sd-host>:8000/sd_inpaint
REMOVE_BG_API_URL=http://<sd-host>:8000/remove_background
```

`<sd-host>` is the LAN hostname or IP of the GPU machine — `localhost` if you're running the bot and SD on the same box. If you change which machine runs SD, **update all four**.

`ANALYZE_IMAGE_API_URL` is in `.env-stable.example` but is not currently consumed by the bot or served by the backend; it is safe to leave at its default. Image vision goes through LM Studio's vision-capable model (set `ENABLE_VISION=true`), not through the SD backend.

### Models, LoRAs, GPU sizing

- **Model selection.** The model id is set as `StableDiffusionConfig.REPO_NAME` near the top of `sd-api/sd_api.py` (default `stabilityai/stable-diffusion-3.5-medium`, with `USE_SDXL = True`). A handful of community model ids are listed as commented-out alternatives. To change models, edit that constant in code — there is currently no env-var override for the model id, and the change requires restarting the SD process.
- **LoRA.** Set `LORA_PATH` (and optionally `LORA_WEIGHT`, default `1.0`) in the SD process's environment. The LoRA is loaded and fused into the pipeline at startup. Switching LoRAs requires a backend restart.
- **GPU sizing.** A 24GB-class GPU (RTX 3090 / 4090 / A5000-class) comfortably runs the default SD 3.5 Medium config with the on-startup `qfloat8` quantization (via `optimum-quanto`) plus `enable_model_cpu_offload()` and attention slicing. Smaller GPUs may work for SD 1.5 or smaller SDXL checkpoints with the same code path — verify by trying it; out-of-memory will surface in the SD host's log.
- **Apple Silicon.** Performance is generally slower than a discrete NVIDIA GPU but workable for casual use. M1/M2/M3 unified memory is shared between CPU and GPU; 32GB+ is recommended. See [`sd-api/M1_MAC_SETUP.md`](../sd-api/M1_MAC_SETUP.md).

### Verify it's up

From the bot host (or anywhere on the LAN that can reach the SD host):

```bash
curl http://<sd-host>:8000/health
# {"status":"ok","model":"stabilityai/stable-diffusion-3.5-medium"}
```

A `200 OK` with the configured model id confirms the pipeline finished loading. From inside Discord, `/status` reports SD-backend reachability alongside the LLM, and `/sd a quick test` is the end-to-end smoke test.
