#!/usr/bin/env bash
# definitive vLLM: Qwen3.8-27B-FP8 TP2 + MTP3 + flashqla_legacy (normal/128K profile)
# 用法: bash start_qwen27b_fp8.sh
set -euo pipefail
export CUDA_HOME=${CUDA_HOME:-/home/landrom-ubuntu/cudatk}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.5}
cd /home/landrom-ubuntu/vLLM-2080Ti-Definitive
exec ./launcher.sh --non-interactive \
  --model-dir /home/landrom-ubuntu/models/Qwen3.8-27B-FP8 \
  --profile qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env \
  --mode normal \
  --gpu-devices 0,1 \
  --service-scope lan \
  --set PORT=8095 \
  --set VLLM_USE_FLASHINFER_SAMPLER=0
