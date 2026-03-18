# Parameter Golf — Approach Notes

## Strategy

Combining quantization-aware training (QAT) with architecture-efficient design to maximize quality within the 16MB constraint.

### 1. QAT over PTQ
Train the model while compressed so it adapts to low-precision weights during training. 2-bit QAT loses ~4% accuracy vs full precision, far better than post-training quantization at the same bitwidth.

### 2. Architecture Efficiency
- Tied input/output embeddings (~40% parameter savings)
- Shared/recycled transformer layers
- Flash Attention
- At 2-bit precision, 16MB fits ~32M parameters
- At 4-bit, ~16M parameters

### 3. Knowledge Distillation
Larger pretrained teacher model guides the small student during training. 8xH100 budget allows running teacher alongside student.

### 4. Training Maximization
- Aggressive sequence packing (multiple examples per input)
- Curriculum data ordering on FineWeb (easier→harder)
- Cosine LR scheduling
- Gradient accumulation to maximize 10 min of 8xH100

### 5. Iterative Compression
Start slightly over budget, train with QAT, prune to fit. Models trained with compression awareness survive pruning better than those compressed after training.

## The Math

| Bitwidth | Parameters in 16MB | Equivalent |
|----------|-------------------|------------|
| 2-bit | ~32M | GPT-2 small |
| 3-bit | ~21M | Between GPT-2 small/medium |
| 4-bit | ~16M | Compact transformer |

## Experiments Planned

- [ ] Run baseline (9-layer, 512-dim, 1024-vocab, tied embeddings)
- [ ] Test 2-bit QAT with AngelSlim/QTIP techniques
- [ ] Test depth recurrence (reuse layers to save parameters)
- [ ] Knowledge distillation from larger model
- [ ] Curriculum data ordering experiments
- [ ] Custom tokenizer optimization (BPE vocab size vs embedding cost tradeoff)
- [ ] Architecture search: width vs depth vs heads allocation

## Background

5 production fine-tuned models (7B-72B) deployed via QLoRA/GGUF/NVFP4 quantization on NVIDIA DGX hardware. Deep experience with the compression-quality tradeoff across bitwidths.

## Status

Awaiting compute credits. Local experimentation with MLX baseline in progress.
