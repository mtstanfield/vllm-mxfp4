# ParoQuant-MXFP4 (rotations on MXFP4 weights) build on RunPod - 2026-09-17

Goal: the fork's `paroquant_mxfp4` format (z-lab's trained rotations + OCP MXFP4 e2m1/e8m0-32 weights, 4.25 bpw, the same
bytes as AMD's MXFP4 -> zero context cost) built with the fork's one-shot `build_hybrid.py`, PLUS a pod-side quality gate
(wikitext PPL of the builder's PSEUDO checkpoint vs the bf16 base) so we know before transferring anything whether it beats
AMD's AWQ-MXFP4 (served delta +2.8% wiki; our home AWQ quants measured +3.0-3.2% on this dequant harness).

## Pod spec (same as custom-quant/RUNBOOK.md)
1x H100 80GB (or A100 80GB), Secure Cloud On-Demand, template "Runpod PyTorch" (CUDA 12.x, sshd), Volume 150 GB at
/workspace, SSH over exposed TCP (direct root@<ip> -p <port>; the ssh.runpod.io proxy cannot scp/rsync).
Disk: bf16 base 55.6 GB + z-lab PARO 18.8 GB + real out ~17 GB + pseudo out ~55 GB (fp16) = ~150 GB. Delete the pseudo after the gate.

## On the pod
```bash
pip install -U "transformers>=5.2" safetensors accelerate datasets huggingface_hub hf_transfer
cd /workspace && git clone https://github.com/z-lab/paroquant src && cd src && pip install -e ".[optim]"   # builds the CUDA rotation kernel
mkdir -p /workspace/models /workspace/q
# from Tower: scp -P <port> /mnt/user/appdata/vllm-radiance/custom-quant/paro-mxfp4-pod/*.py root@<ip>:/workspace/q/
cp /workspace/q/mxfp4_shim.py /workspace/src/paroquant/optim/mxfp4.py      # the fork dev's unpublished helper, ours
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3.8-27B --local-dir /workspace/models/Qwen3.8-27B-bf16
HF_HUB_ENABLE_HF_TRANSFER=1 hf download z-lab/Qwen3.8-27B-PARO --local-dir /workspace/models/Qwen3.8-27B-PARO
python3 -c "from paroquant.kernels.cuda import scaled_pairwise_rotation; from paroquant.optim.mxfp4 import scale_rule; print('imports ok', scale_rule())"
# build (per-tensor on the GPU, streams the 55 GB base shard by shard)
SRC=/workspace/src BASE=/workspace/models/Qwen3.8-27B-bf16 PARO=/workspace/models/Qwen3.8-27B-PARO OUT_REAL=/workspace/models/Qwen3.8-27B-PARO-MXFP4 OUT_PSEUDO=/workspace/models/Qwen3.8-27B-PARO-MXFP4-pseudo python3 /workspace/q/build_hybrid_pod.py 2>&1 | tee /workspace/q/build.log
# quality gate: same harness and 40960 tokens for both; judge the DELTA (AMD AWQ-MXFP4 is about +2.8-3%)
python3 /workspace/q/ppl_dir.py --model /workspace/models/Qwen3.8-27B-bf16
python3 /workspace/q/ppl_dir.py --model /workspace/models/Qwen3.8-27B-PARO-MXFP4-pseudo
```
The real checkpoint has NO mtp.* tensors (the builder skips them: "z-lab ships none"); on Tower we graft our fp8 MTP head with
`custom-quant/graft_mtp.py --head-dtype fp8` and fp8 the lm_head with `fp8_lmhead_sharded.py`, exactly as for int5.

## Transfer to Tower (pull from Tower; the pod cannot reach the LAN)
```bash
ssh root@192.168.88.89 'rsync -a --info=progress2 -e "ssh -p <podport>" root@<podip>:/workspace/models/Qwen3.8-27B-PARO-MXFP4/ /mnt/user/Models/vllm-radiance/Qwen3.8-27B-PARO-MXFP4/'
```
Serving needs the plugin's MXFP4 variant (`paroquant/radiance_paroquant_mxfp4.py` + repo `radiance_mxfp4.py` + the fork's
`radiance_mxfp4_fp8.so` hipcc build) wired into `local/patch_paroquant_install.py` with the same fp8-heads / mixed routing;
that is Tower-side work, independent of the pod.
