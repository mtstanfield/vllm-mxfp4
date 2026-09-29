#!/bin/bash
# setup_pod.sh - one-time pod setup (RunPod "PyTorch" template, H100 80GB, >=128 GB RAM, 200 GB volume at /workspace)
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd /workspace
PIP="pip install -q --break-system-packages"      # PEP 668 image: without the flag pip is a silent no-op
$PIP tilelang   # fla GDN backward on Hopper needs it with triton 3.4 (torch 2.8)
$PIP -U "transformers>=5.2" safetensors accelerate datasets huggingface_hub hf_transfer simple_parsing wandb \
        flash-linear-attention ninja numpy tqdm
[ -d src ] || git clone -q https://github.com/z-lab/paroquant src
(cd src && git checkout -q 9ee635a && $PIP -e ".[optim]")          # the commit optimize_mxfp4.py's patches were written against
mkdir -p models pq2
export HF_HUB_ENABLE_HF_TRANSFER=1
[ -f models/Qwen3.8-27B-bf16/config.json ] || hf download Qwen/Qwen3.8-27B --local-dir models/Qwen3.8-27B-bf16
[ -f models/Qwen3.8-27B-PARO/config.json ] || hf download z-lab/Qwen3.8-27B-PARO --local-dir models/Qwen3.8-27B-PARO
# control: the public GPTQ + AWQ MXFP4 of the same model (RedHatAI, 2026-09-18), scored with the same harness
[ -f models/RedHat-Qwen3.8-27B-MXFP4/config.json ] || hf download RedHatAI/Qwen3.8-27B-MXFP4 --local-dir models/RedHat-Qwen3.8-27B-MXFP4
python3 -c "from paroquant.kernels.cuda import scaled_pairwise_rotation; print('rotation kernel OK')"
python3 -c "import fla; print('flash-linear-attention OK')"
python3 -c "import transformers; print('transformers', transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | sed -n 2p
df -h /workspace | tail -1
