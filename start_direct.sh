#!/usr/bin/env bash
# 直接启动（绕过 launcher 的 overcommit 预检），参数取自 launcher 验证过的 Command
export CUDA_HOME=/home/landrom-ubuntu/cudatk
export TORCH_CUDA_ARCH_LIST=7.5
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SM75_SPEC_SYNC_MODE=safe
export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0
export FLASHINFER_ENABLE_AOT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/landrom-ubuntu/vLLM-2080Ti-Definitive
export PATH=/home/landrom-ubuntu/vllm-venv/bin:$CUDA_HOME/bin:$PATH
cd /home/landrom-ubuntu/vLLM-2080Ti-Definitive
exec /home/landrom-ubuntu/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8095 \
  --model /home/landrom-ubuntu/models/Qwen3.8-27B-FP8 \
  --served-model-name qwen27b-fp8-fp16kv-128K-mtp3-text-only-cu128 \
  --dtype half --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --generation-config vllm --max-model-len 32768 --enable-chunked-prefill \
  --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --quantization fp8 --gpu-memory-utilization 0.88 \
  --mamba-cache-mode align --enable-prefix-caching --enable-prompt-tokens-details \
  --language-model-only --skip-mm-profiling --disable-log-stats \
  --reasoning-parser qwen3 \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}'
