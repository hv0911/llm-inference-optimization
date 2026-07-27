#!/bin/bash
# =============================================================================
# Example curl requests for the vLLM OpenAI-compatible API
# =============================================================================
# Make sure the vLLM server is running first:
#   bash serving/serve_vllm.sh
# =============================================================================

BASE_URL="http://localhost:8000"

echo "============================================================"
echo "  vLLM API — curl Examples"
echo "============================================================"

# --- 1. List available models ---
echo -e "\n--- 1. List Models ---"
curl -s "$BASE_URL/v1/models" | python -m json.tool

# --- 2. Text Completion ---
echo -e "\n--- 2. Text Completion ---"
curl -s "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/base",
    "prompt": "The key advantage of quantized LLM inference is",
    "max_tokens": 100,
    "temperature": 0.7
  }' | python -m json.tool

# --- 3. Chat Completion ---
echo -e "\n--- 3. Chat Completion ---"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/base",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "What is GPTQ quantization?"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }' | python -m json.tool

# --- 4. Streaming Chat Completion ---
echo -e "\n--- 4. Streaming Chat ---"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/base",
    "messages": [
      {"role": "user", "content": "Explain PagedAttention briefly."}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": true
  }'

echo -e "\n\n============================================================"
echo "  Done ✓"
echo "============================================================"
