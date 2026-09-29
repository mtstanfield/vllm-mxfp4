# Custom MXFP4 quant of a Qwen3.8/3.6-27B finetune for vLLM-Radiance (R9700)

Target for the first run: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU`
(bf16 safetensors, 55.6 GB, MTP head + vision tower intact, config identical to base Qwen3.8-27B).

Output: a checkpoint in exactly the layout the fork serves today (Quark OCP-MXFP4 body, W4A4 group 32
e8m0; FP8 e4m3 per-channel MTP head; bf16 lm_head/norms/conv1d/vision), ~19 GB.

## Why a rented GPU
The AWQ statistics pass runs the full bf16 model (55 GB). Tower has ~34 GB free RAM and a 32 GB GPU;
the Windows box has 32 GB RAM. One 80 GB NVIDIA card does the whole job in well under an hour.

Pod spec: 1x H100 80GB (or A100 80GB), >=64 GB RAM, >=150 GB disk, a CUDA torch image, ssh access.
Budget: ~1 GPU-hour. Rough cost $2-4.

## RunPod configuration (first-time checklist)
1. Account → Settings → **SSH Public Keys**: paste the Tower root key (`cat /root/.ssh/id_ed25519.pub` on Tower;
   Tower is the box that will rsync the result back, so its key must be there) and optionally the Windows key.
2. Pods → Deploy: **Secure Cloud**, **On-Demand** (not Spot: an interruption mid-quantize wastes the hour).
   GPU: **H100 PCIe 80GB** (or A100 80GB / H100 SXM if cheaper in the region). 1 GPU.
3. Template: **Runpod PyTorch** (official; runs sshd, has torch + CUDA). Set **Volume Disk = 150 GB**
   (persistent, mounts at `/workspace`), Container Disk 30 GB is fine.
4. Expose TCP port **22** in the template's exposed TCP ports so *direct* ssh exists. The default
   `ssh ...@ssh.runpod.io` proxy does NOT do scp/rsync; the direct one (`root@<ip> -p <port>`, shown in the
   Pod's **Connect** tab under **SSH over exposed TCP**) does.
5. Deploy, wait for Running, copy the direct ssh line from Connect.
Everything below lives in `/workspace` (the volume); nothing else on the pod survives a stop.

## On the pod
```bash
pip install -U "transformers>=5.2" safetensors accelerate datasets huggingface_hub hf_transfer
mkdir -p /workspace/q && cd /workspace/q
# scripts: scp this folder's *.py + ref-config.json here (from Tower: scp -P <port> /mnt/user/appdata/vllm-radiance/custom-quant/* root@<ip>:/workspace/q/)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU --local-dir /workspace/q/src
# 1. activation statistics (bf16 model on the GPU, ~5 min incl. load; 128 samples x 2048 tokens)
python3 calib_stats.py --model /workspace/q/src --out /workspace/q/stats.pt --samples 128 --seq-len 2048
# 2. quantize (streams the shards, GPU alpha search; ~10-15 min)
python3 quantize_mxfp4_qwen35.py --src /workspace/q/src --stats /workspace/q/stats.pt --out /workspace/q/out --ref-config /workspace/q/ref-config.json
# 3. sanity: perplexity delta bf16 -> dequantized mxfp4 (~10 min). Judge the DELTA; expect a few %.
python3 verify_dequant_ppl.py --src /workspace/q/src --quant /workspace/q/out
```
Read `out/quant-report.json`: mean relative weight error (the fork measured ~0.116 for plain MXFP4 RTN on the
drafter; AWQ-scaled body layers should land below that), worst tensors, chosen alphas (0 = no scaling helped,
1 = full activation weighting).

## Transfer to Tower (pull from Tower; the pod cannot reach the LAN)
```bash
ssh root@192.168.88.89 'rsync -a --info=progress2 -e "ssh -p <podport>" root@<podip>:/workspace/q/out/ /mnt/user/Models/vllm-radiance/Qwen3.8-27B-DAU-TURBO-MXFP4-mtpfp8/'
```
(19 GB; alternatively `hf upload` to a private repo from the pod and `hf download` on Tower.)

## Serve on Tower (test deployment first, then swap prod)
The serve script picks the checkpoint from `SNAP`; the deploy wrapper exports `MODELS=/mnt/user/Models/vllm-radiance`.
Same architecture and sizes => the measured KV pin for the prod checkpoint applies unchanged.
```bash
# with the GPU free (stops prod: the wrapper does docker rm -f on vllm-qwen38)
ssh root@192.168.88.89 'SNAP=/mnt/user/Models/vllm-radiance/Qwen3.8-27B-DAU-TURBO-MXFP4-mtpfp8 /mnt/user/appdata/llama-gemma31b/deploy-vllm-qwen38.sh' | tail -3
```
Then, in order: `/health` + a chat completion; MTP acceptance from the server log / metrics (prod is 2.6-2.8 accepted
per step with the fp8 head); NIAH 8/8 with the existing harness; the user's own quality bar (Fable-711 lesson: probes != bar).
Rollback = the same command without `SNAP`.

## Sampling
DavidAU cards usually recommend temp ~0.6-1.0 with rep-penalty ~1.05-1.1; the server's override-generation-config
currently pins temp 1.0 / top_p 0.95 / top_k 20 for the base model. Check the card and pass a different
`GEN_CONFIG` to the deploy wrapper if the finetune wants it (MTP acceptance moves with temperature).

## Run log
- **2026-09-09, DavidAU TURBO-Fable-Cold-Fusion 3.8-27B, RunPod H100 PCIe Secure (~1 h, ~$3).**
  stats: 193 samples / 293k tokens (77 ultrachat, 58 wikitext, 58 code). quantize: 496 linears, 128 s,
  mean rel err 0.1205. **PPL wikitext-2 test, 41k tokens: bf16 6.238 -> mxfp4 6.437 (+3.2%).**
  Checkpoint: `/mnt/user/Models/vllm-radiance/Qwen3.8-27B-DAU-TURBO-MXFP4-mtpfp8` (checksum-verified rsync).
- Lessons baked into the scripts:
  * Qwen3.5's decoder RMSNorm is ZERO-CENTERED (`x_norm * (1 + w)`): the AWQ fold is `(1+w)/s - 1`.
    The first pass used `w/s` and scored PPL 2.7 million. `fold_norm()` now does it right.
  * datasets 5: `wikitext` -> `Salesforce/wikitext`; `the-stack-smol` and `tiny-codes` are gated;
    script-based datasets (`github-code-clean`) no longer load. Code now comes from
    `code-search-net/code_search_net` + `m-a-p/CodeFeedback-Filtered-Instruction`.
  * Pod ops: `pip install --break-system-packages` (PEP 668 image); detach with `nohup ... < /dev/null &`
    or the ssh session hangs; `/workspace` is a network volume (~500 MB/s), fine for this.
  * The 12-file bf16 download is ~2 min on the pod; the whole pipeline is <15 min of GPU time.

## Files
- `calib_stats.py` - hooks q_proj / in_proj_qkv / gate_proj / down_proj inputs, writes mean |x| per channel
- `quantize_mxfp4_qwen35.py` - AWQ alpha search + fold + MXFP4 pack for all 64 layers, FP8 MTP head, Quark config
- `verify_dequant_ppl.py` - PPL of bf16 vs dequantized quant on wikitext-2 test
- `ref-config.json` - the production checkpoint's config.json (quantization_config copied verbatim)
- provenance: `/mnt/user/appdata/vllm-radiance/quantize_dflash_mxfp4.py` (the drafter quantizer this is ported from)
  and `fp8_mtp.py` (the MTP head recipe)
