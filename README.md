# buildGPT — Built from Scratch

> A GPT language model implemented entirely from first principles, built by completing the [NeetCode ML Course](https://neetcode.io/practice?tab=coreSkills&topic=Machine+Learning).
> Every file in this repo is original code written and submitted as part of the course — from gradient descent all the way up to a working transformer.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Implementation Highlights](#implementation-highlights)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Learning Path](#learning-path)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Overview

This project is a **GPT (Generative Pre-trained Transformer) built entirely from scratch** — no HuggingFace, no pre-built transformer libraries. Every component, from the neuron-level backpropagation to the multi-head attention mechanism, was designed, implemented, and tested through the NeetCode ML course problem set.

The model can be trained on any character-level text corpus and used to generate new text autoregressively.

> **Proof of concept:** The original Colab notebook from the course uses the model to generate new Drake lyrics after training on his discography.

---

## Architecture

The model follows the original GPT (decoder-only transformer) design:

```
Input Tokens
     │
     ▼
Token Embeddings + Positional Encoding
     │
     ▼
┌─────────────────────────────────┐
│         Transformer Block       │  ×N layers
│  ┌───────────────────────────┐  │
│  │  Layer Normalization      │  │
│  │  Multi-Head Self-Attention│  │
│  │  (+ KV-Cache for infer.) │  │
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │  Layer Normalization      │  │
│  │  Feed-Forward MLP         │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
     │
     ▼
Final Layer Normalization
     │
     ▼
Linear Projection → Logits (Vocab Size)
```

**Attention variants implemented:**
- Single-head self-attention (`attention.py`)
- Multi-head attention (`multi_head_attention.py`)
- Grouped-query attention (`grouped_query_attention.py`) — the variant used in LLaMA 2/3 and Mistral

---

## Project Structure

```
neetcode-gpt/
│
├── train.py                      # GPT training loop (AdamW + cross-entropy)
├── generate.py                   # Autoregressive text generation
├── requirements.txt              # Dependencies
│
├── model/                        # Full GPT architecture
│   ├── gpt.py                    # Top-level GPT model
│   ├── transformer.py            # Transformer block (attn + FFN + residual)
│   ├── attention.py              # Scaled dot-product self-attention (single head)
│   ├── multi_head_attention.py   # Multi-head attention
│   ├── grouped_query_attention.py# GQA — efficient attention for inference
│   ├── kv_cache.py               # Key-Value cache for fast autoregressive generation
│   ├── embeddings.py             # Token embedding layer
│   ├── positional_encoding.py    # Sinusoidal positional encoding
│   ├── normalization.py          # Layer normalization
│   ├── batch_normalization.py    # Batch normalization
│   └── rms_normalization.py      # RMS normalization (used in modern LLMs)
│
├── data/                         # Data pipeline
│   ├── tokenizer.py              # BPE (Byte-Pair Encoding) tokenizer
│   ├── vocab.py                  # Character-level vocabulary builder
│   ├── loader.py                 # Batched training data loader
│   ├── dataset.py                # GPT dataset preparation
│   ├── nlp_preprocessing.py      # Text cleaning & NLP preprocessing
│   └── tokenizer_utils.py        # Tokenization edge case handling
│
└── foundations/                  # Neural network primitives from scratch
    ├── neuron.py                 # Single neuron with weights and bias
    ├── backprop.py               # Manual backpropagation
    ├── mlp.py                    # Multi-layer perceptron
    ├── activations.py            # ReLU, sigmoid, tanh, GELU
    ├── loss.py                   # MSE, cross-entropy loss
    ├── training_loop.py          # Generic training loop
    └── dead_relu_detector.py     # Diagnostic: detect dead ReLU neurons
```

---

## Implementation Highlights

### Training Loop (`train.py`)

Uses **AdamW** optimizer with a cross-entropy loss over next-token predictions. Each epoch seeds the random number generator for reproducibility, samples random context windows, and does a full forward-backward-update pass.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
for epoch in range(epochs):
    torch.manual_seed(epoch)
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + context_length] for i in ix])
    logits = model(x)
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Text Generation (`generate.py`)

Autoregressive sampling with context windowing. At each step:
1. Crop context to `context_length` if it exceeds it
2. Run a forward pass, take the last token's logits
3. Apply softmax → sample the next token via `torch.multinomial`
4. Append token to the running context and decode to character

### KV-Cache (`model/kv_cache.py`)

Implements a key-value cache so that during inference, previously computed keys and values are reused rather than recomputed — reducing the per-step compute from O(T²) to O(T) after the first pass.

### Grouped-Query Attention (`model/grouped_query_attention.py`)

An efficient variant of multi-head attention where multiple query heads share a single key/value head. This reduces the memory footprint of the KV cache during inference — the technique used in LLaMA 2, Mistral, and Gemma.

### Normalization Variants

Three normalization strategies are implemented:
- **Layer Norm** — standard Transformer normalization
- **Batch Norm** — classic deep learning normalization (implemented from scratch for understanding)
- **RMS Norm** — lighter alternative used in modern LLMs (LLaMA, GPT-NeoX)

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Raghav0079/neetcode-gpt.git
cd neetcode-gpt
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0.0 | Model, tensors, autograd |
| `numpy` | ≥ 1.24.0 | Numerical utilities |
| `torchtyping` | ≥ 0.1.4 | Type-annotated tensor shapes |

---

## Usage

### Train the model

```bash
python train.py
```

The training loop returns the final cross-entropy loss rounded to 4 decimal places.

### Generate text

```bash
python generate.py
```

The generator runs autoregressive sampling and prints the generated text. You can control:
- `new_chars` — how many characters to generate
- `context` — the seed token sequence
- `context_length` — the model's maximum context window
- `int_to_char` — the vocabulary mapping

---

## Learning Path

This project was built by progressively completing the NeetCode ML course modules. Each module builds on the last:

| Stage | Topics | Files |
|---|---|---|
| **1. Math Foundations** | Gradient descent, activations, loss functions | `foundations/activations.py`, `foundations/loss.py` |
| **2. Neural Networks** | Neurons, backprop, MLPs | `foundations/neuron.py`, `foundations/backprop.py`, `foundations/mlp.py` |
| **3. PyTorch** | Tensors, autograd, modules | Used throughout `model/` |
| **4. NLP Pipeline** | Tokenization, embeddings, vocab | `data/` |
| **5. Attention** | Self-attention, multi-head, GQA | `model/attention.py`, `model/multi_head_attention.py` |
| **6. Transformer** | Full block with normalization + residuals | `model/transformer.py` |
| **7. GPT** | Assembled model + training + generation | `model/gpt.py`, `train.py`, `generate.py` |

---

## Tech Stack

- **Language:** Python 3.8+
- **Framework:** PyTorch 2.0+
- **Paradigm:** From-scratch implementation (no transformer libraries)
- **Architecture:** Decoder-only GPT (autoregressive language model)
- **Tokenization:** Character-level + BPE

---

## Author

**Raghav Mishra**
Built as part of the [NeetCode ML Course](https://neetcode.io/practice?tab=coreSkills&topic=Machine+Learning) — May 2026

> This repo represents a complete bottom-up understanding of how GPT-style models work, from the math of a single neuron all the way to autoregressive text generation.
