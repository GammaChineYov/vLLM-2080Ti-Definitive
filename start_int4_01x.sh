#!/usr/bin/env bash
export CUDA_HOME=/home/landrom-ubuntu/cudatk
export TORCH_CUDA_ARCH_LIST=7.5
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_ENABLE_AOT=1
export VLLM_SM75_SPEC_SYNC_MODE=safe
export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0
export PYTHONPATH=/home/landrom-ubuntu/vLLM-2080Ti-Definitive
export PATH=/home/landrom-ubuntu/vllm-venv/bin:$CUDA_HOME/bin:$PATH
cd /home/landrom-ubuntu/vLLM-2080Ti-Definitive
exec /home/landrom-ubuntu/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8095 \
  --model /media/landrom-ubuntu/新加卷/llm-deploy/models/Qwen3.8-27B-AWQ-INT4 \
  --served-model-name qwen27b-local \
  --dtype half --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --max-model-len 262144 --enable-chunked-prefill \
  --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --quantization compressed-tensors --gpu-memory-utilization 0.90 \
  --mamba-cache-mode align --enable-prefix-caching --enable-prompt-tokens-details \
  --language-model-only --skip-mm-profiling --disable-log-stats \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}' \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
