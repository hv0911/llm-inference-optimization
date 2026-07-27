# Quantization: Theory & Implementation

A deep-dive into weight quantization for Large Language Models, covering the theory behind AWQ and GPTQ, and how they are implemented in this repository.

---

## Table of Contents

1. [What is Quantization?](#what-is-quantization)
2. [Weight-Only Quantization](#weight-only-quantization)
3. [AWQ: Activation-Aware Weight Quantization](#awq-activation-aware-weight-quantization)
4. [GPTQ: Post-Training Quantization](#gptq-post-training-quantization)
5. [AWQ vs GPTQ Comparison](#awq-vs-gptq-comparison)
6. [Calibration](#calibration)
7. [Group Quantization](#group-quantization)
8. [Activation Quantization (W8A8)](#activation-quantization)
9. [Implementation Details](#implementation-details)

---

## What is Quantization?

Quantization reduces the precision of model weights from higher-bit representations (FP16/BF16, 16 bits) to lower-bit representations (INT4, 4 bits). This achieves:

- **Reduced memory footprint**: ~3.5–4x reduction for INT4 vs FP16
- **Faster inference**: Smaller weights = faster memory transfers (LLM inference is memory-bandwidth bound)
- **Lower cost**: Serve larger models on smaller GPUs

The key challenge is minimizing the accuracy degradation caused by reduced precision.

### Quantization Formula

For uniform symmetric quantization:

```
Q(w) = clamp(round(w / s), -2^(b-1), 2^(b-1) - 1)
```

Where:
- `w` = original weight
- `s` = scale factor = max(|w|) / (2^(b-1) - 1)
- `b` = number of bits
- `Q(w)` = quantized weight (integer)

Dequantization (at inference): `w_approx = Q(w) × s`

---

## Weight-Only Quantization

In weight-only quantization (W4A16), only the model weights are quantized to INT4, while activations remain in FP16/BF16. This is the most common approach for LLM serving because:

1. **Weights dominate memory**: For a 0.6B model, weights are ~1.2GB in FP16
2. **Activations are dynamic**: They vary per input, making static quantization less effective
3. **Hardware support**: Modern GPU kernels (Marlin, Machete) are optimized for W4A16

```
Inference: output = Dequant(W_int4) × X_fp16
                    ↑                  ↑
              4-bit weights       16-bit activations
```

---

## AWQ: Activation-Aware Weight Quantization

**Paper**: *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (Lin et al., 2023)

### Core Insight

Not all weights are equally important. AWQ observes that a small fraction of weight channels (typically 1%) are **salient** — they correspond to large activation magnitudes. Quantization errors in these channels cause disproportionate output errors.

### Algorithm

1. **Observe activations**: Run calibration data through the model and record activation magnitudes per channel
2. **Identify salient channels**: Find channels with the highest average activation values
3. **Scale salient weights**: Multiply salient weights by a scaling factor `s` before quantization, which effectively increases their precision
4. **Compensate**: Divide the corresponding activations by `s` to maintain mathematical equivalence

```
Original:    output = W × X
After AWQ:   output = (W × diag(s)) × (diag(1/s) × X)
                      ↑ quantized with higher effective precision
```

### Why It Works

By scaling salient weights up before quantization:
- They occupy more of the INT4 range, reducing relative quantization error
- The inverse scaling on activations is exact (FP16 precision)
- Net effect: the important weights are quantized more carefully

### Implementation in This Repo

```python
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = [
    AWQModifier(),                    # Step 1-3: Scale computation
    QuantizationModifier(             # Step 4: Apply quantization
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    ),
]
```

---

## GPTQ: Post-Training Quantization

**Paper**: *GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers* (Frantar et al., 2022)

### Core Insight

GPTQ frames quantization as a layer-wise optimization problem. For each layer, it finds the quantized weights that minimize the output reconstruction error, using second-order (Hessian) information.

### Algorithm

Based on the Optimal Brain Surgeon (OBS) framework:

1. **Compute Hessian**: For each layer, compute `H = 2 × X^T × X` using calibration data
2. **Process columns**: For each weight column:
   a. Quantize the column using rounding
   b. Compute the quantization error
   c. Distribute the error to remaining unprocessed columns using the Hessian inverse
3. **Result**: Quantized weights that account for inter-column correlations

```
For layer l:
  min_Q  || W_l × X_l - Q(W_l) × X_l ||^2
  subject to Q(W_l) ∈ INT4
```

### Key Innovation: Lazy Batch Updates

GPTQ doesn't update all remaining columns after each quantization. Instead, it processes columns in blocks (typically 128) and applies batched updates, reducing computational cost from O(d³) to O(d × block² + d² × block).

### Implementation in This Repo

```python
from llmcompressor.modifiers.quantization import GPTQModifier

recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
)
```

---

## AWQ vs GPTQ Comparison

| Aspect | AWQ | GPTQ |
| --- | --- | --- |
| **Approach** | Activation-based scaling | Hessian-based error correction |
| **Optimization** | Protects salient channels | Minimizes layer-wise reconstruction error |
| **Speed** | Faster calibration | Slower (Hessian computation) |
| **Accuracy** | Generally better at same bit-width | Slightly lower, but very competitive |
| **Memory during quant** | Moderate | Higher (stores Hessian) |
| **Inference speed** | Same (both use Marlin kernels in vLLM) | Same |
| **Best for** | Production deployment | Broad compatibility |

---

## Calibration

Both methods require a small calibration dataset (typically 128–512 examples) to:
- AWQ: Compute activation statistics for channel saliency
- GPTQ: Compute the Hessian matrix for error correction

### Calibration Dataset Choice

This repo uses **WikiText-2** (raw) by default — a general-purpose English text corpus. For domain-specific models, using domain-relevant calibration data can improve quantized model quality.

```yaml
# config/default.yaml
quantization:
  calibration_dataset: "wikitext"
  calibration_dataset_config: "wikitext-2-raw-v1"
  num_calibration_samples: 256
```

---

## Group Quantization

Rather than using a single scale factor per entire weight tensor, group quantization uses one scale per group of weights (e.g., 128 consecutive values). This provides:

- **Better accuracy**: Each group has its own scale, reducing quantization error
- **Minimal overhead**: Scale factors are small compared to weight matrices
- **Standard practice**: Group size 128 is the de facto standard for INT4

```
Tensor: [w1, w2, ..., w128 | w129, ..., w256 | ...]
Scales: [    s1            |      s2           | ...]
         ← group_size=128 →
```

---

## Activation Quantization

While this repo focuses on **weight-only (W4A16)** quantization, activation quantization (W8A8) is also available:

| Method | Weights | Activations | Use Case |
| --- | --- | --- | --- |
| W4A16 | INT4 | FP16 | Memory-constrained, standard serving |
| W8A8 | INT8 | INT8 | Throughput-optimized, Hopper/Blackwell GPUs |
| FP8 | FP8 | FP8 | Modern hardware with native FP8 support |

W8A8 quantization requires dynamic activation scaling at inference time and hardware that supports INT8 matrix multiply (all modern NVIDIA GPUs do).

---

## Implementation Details

### llm-compressor Architecture

```
llm-compressor
├── oneshot()           # Main entry point
├── Modifiers/
│   ├── AWQModifier     # Computes activation-aware scales
│   ├── GPTQModifier    # Hessian-based weight optimization
│   └── QuantizationModifier  # Applies quantization scheme
└── Output: compressed-tensors format
    ├── quantized weights (INT4)
    ├── scale factors (FP16)
    ├── zero points (optional)
    └── config.json (quantization metadata)
```

### Compressed Tensors Format

The output uses the `compressed-tensors` format, which vLLM natively understands:

```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "config_groups": {
      "group_0": {
        "targets": ["Linear"],
        "weights": {
          "num_bits": 4,
          "strategy": "group",
          "group_size": 128
        }
      }
    }
  }
}
```

vLLM automatically selects the optimal kernel (Marlin for INT4) when loading a `compressed-tensors` model, providing near-native inference performance.
