# =============================================================================
# Dockerfile — Production LLM Optimization
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder): Install quantization and evaluation dependencies
#   Stage 2 (runtime): Lean vLLM serving image
#
# Build:
#   docker build -t llm-optimization .
#   docker build --target runtime -t llm-optimization-serve .
#
# Run (serving):
#   docker run --gpus all -p 8000:8000 -v ./models:/app/models \
#     llm-optimization-serve --model /app/models/awq
# =============================================================================

# ---- Stage 1: Builder (full pipeline) ----
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# ---- Stage 2: Runtime (lean serving) ----
FROM vllm/vllm-openai:latest AS runtime

WORKDIR /app

# Copy only serving-related files
COPY serving/ ./serving/
COPY config/ ./config/

# Default: serve base model
ENV MODEL_PATH="/app/models/base"
ENV HOST="0.0.0.0"
ENV PORT="8000"
ENV GPU_MEMORY_UTILIZATION="0.85"
ENV MAX_MODEL_LEN="2048"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["vllm", "serve"]
CMD ["${MODEL_PATH}", \
     "--host", "${HOST}", \
     "--port", "${PORT}", \
     "--gpu-memory-utilization", "${GPU_MEMORY_UTILIZATION}", \
     "--max-model-len", "${MAX_MODEL_LEN}", \
     "--trust-remote-code"]
