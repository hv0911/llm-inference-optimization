#!/bin/bash
# =============================================================================
# vLLM Model Server — Launch Script
# =============================================================================
# Starts a vLLM OpenAI-compatible API server for the specified model.
#
# Usage:
#   bash serving/serve_vllm.sh                          # Serve base model
#   bash serving/serve_vllm.sh --variant awq             # Serve AWQ model
#   bash serving/serve_vllm.sh --variant gptq            # Serve GPTQ model
#   bash serving/serve_vllm.sh --model /path/to/model    # Custom model path
#
# The server provides an OpenAI-compatible API at http://localhost:8000
# =============================================================================

set -euo pipefail

# --- Defaults ---
VARIANT="base"
MODEL_PATH=""
HOST="0.0.0.0"
PORT="8000"
GPU_MEMORY_UTILIZATION="0.85"
MAX_MODEL_LEN="2048"
TENSOR_PARALLEL_SIZE="1"
DTYPE="auto"
ENABLE_PREFIX_CACHING=""
EXTRA_ARGS=""

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)
            VARIANT="$2"; shift 2 ;;
        --model)
            MODEL_PATH="$2"; shift 2 ;;
        --host)
            HOST="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"; shift 2 ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --dtype)
            DTYPE="$2"; shift 2 ;;
        --enable-prefix-caching)
            ENABLE_PREFIX_CACHING="--enable-prefix-caching"; shift ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# --- Resolve Model Path ---
if [ -z "$MODEL_PATH" ]; then
    case $VARIANT in
        base|fp16)
            MODEL_PATH="models/base" ;;
        awq)
            MODEL_PATH="models/awq" ;;
        gptq)
            MODEL_PATH="models/gptq" ;;
        *)
            echo "Unknown variant: $VARIANT"
            echo "Valid variants: base, awq, gptq"
            exit 1 ;;
    esac
fi

# --- Print Configuration ---
echo "============================================================"
echo "  vLLM Server Configuration"
echo "============================================================"
echo "  Model:                 $MODEL_PATH"
echo "  Variant:               $VARIANT"
echo "  Host:                  $HOST"
echo "  Port:                  $PORT"
echo "  GPU Memory Util:       $GPU_MEMORY_UTILIZATION"
echo "  Max Model Len:         $MAX_MODEL_LEN"
echo "  Tensor Parallel:       $TENSOR_PARALLEL_SIZE"
echo "  Dtype:                 $DTYPE"
echo "  Prefix Caching:        ${ENABLE_PREFIX_CACHING:-disabled}"
echo "============================================================"
echo ""
echo "  API will be available at: http://${HOST}:${PORT}/v1"
echo "  Docs: http://${HOST}:${PORT}/docs"
echo ""
echo "============================================================"

# --- Launch vLLM ---
exec vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    $ENABLE_PREFIX_CACHING \
    $EXTRA_ARGS
