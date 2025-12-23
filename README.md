<p align="center">
  <h1 align="center">🔬 GPT-2 Architecture: A Deep Learning Study</h1>
  <p align="center">
    <strong>Educational Implementation of the GPT-2 Language Model from Scratch</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.9-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Status-Educational-green?style=for-the-badge" alt="Status">
    <img src="https://img.shields.io/badge/Focus-Learning-blue?style=for-the-badge" alt="Focus">
  </p>
</p>

---

## 📋 Table of Contents

- [About The Project](#-about-the-project)
- [Important Disclaimer](#-important-disclaimer)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Technical Implementation](#-technical-implementation)
- [PyTorch Modules Deep Dive](#-pytorch-modules-deep-dive)
- [Techniques & Methods](#-techniques--methods)
- [Getting Started](#-getting-started)
- [Learning Curriculum](#-learning-curriculum)
- [References](#-references)

---

## 🎯 About The Project

This repository serves as a comprehensive educational resource for understanding the inner workings of **GPT-2 (Generative Pre-trained Transformer 2)**, one of the foundational architectures in modern Natural Language Processing.

The primary objective is to provide a **hands-on, from-scratch implementation** that demystifies transformer-based language models. Each component is implemented individually with detailed explanations, making this an ideal resource for students, researchers, and practitioners seeking to deepen their understanding of LLM architectures.

### Key Learning Outcomes

- ✅ Understanding of transformer architecture fundamentals
- ✅ Implementation of self-attention mechanisms
- ✅ Mastery of positional and token embeddings
- ✅ Knowledge of training pipelines for language models
- ✅ Practical experience with PyTorch neural network modules

---

## ⚠️ Important Disclaimer

> **Note:** This is an **educational implementation** designed for learning purposes.

### Training Scope

| Aspect | This Implementation | Production GPT-2 |
|--------|---------------------|------------------|
| **Dataset Size** | ~20KB - 17MB (small text files) | 40GB+ (WebText) |
| **Training Data** | "The Verdict" short story, TinyStories sample | Millions of web pages |
| **Training Duration** | Minutes to hours | Days to weeks on GPU clusters |
| **Model Purpose** | Understanding architecture | Production text generation |
| **Hardware** | Consumer CPU/GPU | TPU/GPU clusters |

**We intentionally trained on minimal data** (`the-verdict.txt` ~20KB and `tinystories_20k.txt` ~17MB) to focus on understanding the architecture and training pipeline rather than achieving production-quality results. This approach allows for rapid experimentation and learning without requiring extensive computational resources.

---

## 🏛️ Model Architecture

### GPT-2 124M Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│                        GPT-2 ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌──────────────────────────────────────────────────────┐     │
│    │                  INPUT EMBEDDINGS                     │     │
│    │  ┌─────────────────┐    ┌─────────────────────────┐  │     │
│    │  │ Token Embedding │ +  │ Positional Embedding    │  │     │
│    │  │   (50257×768)   │    │      (1024×768)         │  │     │
│    │  └─────────────────┘    └─────────────────────────┘  │     │
│    └──────────────────────────────────────────────────────┘     │
│                              ↓                                   │
│    ┌──────────────────────────────────────────────────────┐     │
│    │              TRANSFORMER BLOCK (×12)                  │     │
│    │  ┌────────────────────────────────────────────────┐  │     │
│    │  │              Layer Normalization                │  │     │
│    │  └────────────────────────────────────────────────┘  │     │
│    │                         ↓                            │     │
│    │  ┌────────────────────────────────────────────────┐  │     │
│    │  │         Multi-Head Self-Attention              │  │     │
│    │  │    (12 heads × 64 dim = 768 total dim)         │  │     │
│    │  └────────────────────────────────────────────────┘  │     │
│    │                    ↓ + Residual                      │     │
│    │  ┌────────────────────────────────────────────────┐  │     │
│    │  │              Layer Normalization                │  │     │
│    │  └────────────────────────────────────────────────┘  │     │
│    │                         ↓                            │     │
│    │  ┌────────────────────────────────────────────────┐  │     │
│    │  │            Feed-Forward Network                 │  │     │
│    │  │        (768 → 3072 → 768) + GELU               │  │     │
│    │  └────────────────────────────────────────────────┘  │     │
│    │                    ↓ + Residual                      │     │
│    └──────────────────────────────────────────────────────┘     │
│                              ↓                                   │
│    ┌──────────────────────────────────────────────────────┐     │
│    │              FINAL LAYER NORMALIZATION               │     │
│    └──────────────────────────────────────────────────────┘     │
│                              ↓                                   │
│    ┌──────────────────────────────────────────────────────┐     │
│    │              OUTPUT PROJECTION (768→50257)           │     │
│    │                   (Vocabulary Logits)                │     │
│    └──────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 50,257 | BPE vocabulary size (GPT-2 standard) |
| `context_length` | 1,024 | Maximum sequence length |
| `emb_dim` | 768 | Embedding dimension |
| `n_heads` | 12 | Number of attention heads |
| `n_layers` | 12 | Number of transformer blocks |
| `drop_rate` | 0.1 | Dropout probability |
| `qkv_bias` | False | No bias in Q, K, V projections |

---

## 📁 Project Structure

```
GPT-2/
│
├── 📓 Core Implementation
│   ├── gpt-2_all_entire_pipeline.ipynb    # Complete training pipeline
│   ├── EntireGPT_architecture.ipynb       # Full architecture implementation
│   └── multiheadAttention.ipynb           # Multi-head attention deep dive
│
├── 📓 Practice Notebooks
│   └── practice/
│       ├── BYTEPAIRtokenizer.ipynb        # BPE tokenization study
│       ├── TOKENEMBEDINGS.ipynb           # Embedding mechanics
│       ├── 1.ipynb                        # Fundamentals
│       ├── 2.ipynb                        # Intermediate concepts
│       └── practice.ipynb                 # Experimentation
│
├── 📂 Data
│   └── datasets/
│       ├── the-verdict.txt                # Primary training text (~20KB)
│       └── tinystories_20k.txt            # Extended corpus (~17MB)
│
├── 📊 Outputs
│   └── loss-plot.pdf                      # Training loss visualization
│
├── 📄 Configuration
│   ├── requirements.txt                   # Dependencies
│   └── .gitignore                         # Git ignore rules
│
└── 📂 Environment
    └── venv/                              # Python virtual environment
```

---

## 🔧 Technical Implementation

### Core Components Implemented

```python
# Configuration Dictionary
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

### 1. Layer Normalization

```python
class LayerNorm(nn.Module):
    """
    Implements Layer Normalization for stabilizing training.
    Applied BEFORE attention and FFN (Pre-LayerNorm variant).
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

### 2. GELU Activation Function

```python
class GELU(nn.Module):
    """
    Gaussian Error Linear Unit - smoother alternative to ReLU.
    Used in GPT-2's feed-forward networks.
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

### 3. Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """
    Implements scaled dot-product attention with causal masking.
    Splits attention into multiple parallel heads for richer representations.
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask as buffer (not a parameter)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
```

---

## 🧰 PyTorch Modules Deep Dive

### Modules Used & Rationale

| Module | Purpose | Why We Used It |
|--------|---------|----------------|
| `nn.Module` | Base class for all neural network modules | Provides parameter management, device handling, and forward/backward hooks |
| `nn.Linear` | Fully connected layer | Implements W×x+b transformations for Q, K, V projections and FFN layers |
| `nn.Embedding` | Lookup table for embeddings | Efficiently maps discrete tokens to continuous vectors |
| `nn.Dropout` | Regularization | Prevents overfitting by randomly zeroing elements during training |
| `nn.Parameter` | Learnable parameter | Wraps tensors as trainable parameters (used in LayerNorm) |
| `nn.Sequential` | Container module | Chains multiple layers into a single callable unit |

### Special PyTorch Features

#### `register_buffer()`

```python
self.register_buffer("mask", torch.triu(...))
```

**Why:** The causal mask is a fixed tensor that:
- ❌ Should NOT be updated during backpropagation
- ✅ Should move to the same device as the model (CPU/GPU)
- ✅ Should be saved with model state dict

#### Tensor Operations Used

| Operation | Usage | Purpose |
|-----------|-------|---------|
| `@` (matmul) | `queries @ keys.T` | Compute attention scores efficiently |
| `.view()` | Reshape tensors | Split/merge attention heads |
| `.transpose()` | Swap dimensions | Rearrange for batch matrix multiplication |
| `.contiguous()` | Memory layout | Ensure tensor is stored contiguously after transpose |
| `.masked_fill_()` | In-place masking | Apply causal mask with -inf values |

---

## 🔬 Techniques & Methods

### 1. Tokenization Strategy

| Technique | Implementation | Benefit |
|-----------|----------------|---------|
| **Byte Pair Encoding (BPE)** | `tiktoken.get_encoding("gpt2")` | Subword tokenization balances vocabulary size and sequence length |

### 2. Attention Mechanism

| Technique | Description |
|-----------|-------------|
| **Scaled Dot-Product Attention** | Divides attention scores by √d_k to prevent softmax saturation |
| **Causal Masking** | Uses upper triangular mask with -∞ to prevent attending to future tokens |
| **Multi-Head Parallelization** | Splits attention into 12 heads, each learning different aspects |

### 3. Normalization Approach

| Technique | Position | Rationale |
|-----------|----------|-----------|
| **Pre-LayerNorm** | Before attention/FFN | Better gradient flow, more stable training than Post-LayerNorm |

### 4. Activation Function

| Technique | Formula | Why |
|-----------|---------|-----|
| **GELU** | `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` | Smoother than ReLU, enables non-zero gradients for negative inputs |

### 5. Regularization

| Technique | Rate | Application |
|-----------|------|-------------|
| **Dropout** | 0.1 | Applied after attention weights and residual connections |

### 6. Residual Connections

| Technique | Implementation |
|-----------|----------------|
| **Skip Connections** | `x = x + sublayer(x)` enables gradient flow through deep networks |

### 7. Data Pipeline

| Technique | Purpose |
|-----------|---------|
| **Sliding Window** | Creates overlapping training samples with configurable stride |
| **DataLoader** | Batches data, shuffles, and handles efficient loading |

### 8. Text Generation Strategies

#### Temperature Sampling

**Temperature** controls the randomness of predictions by scaling the logits before applying softmax.

```python
logits = logits / temperature
probs = torch.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| **0.1 - 0.3** | Very focused, deterministic | Factual responses, code |
| **0.5 - 0.7** | Balanced creativity | General text generation |
| **0.8 - 1.0** | More diverse output | Creative writing |
| **1.0 - 2.0** | Highly random | Brainstorming, exploration |

**How it works:**
- **T < 1.0**: Sharpens probability distribution → high-probability tokens dominate
- **T = 1.0**: Original distribution unchanged
- **T > 1.0**: Flattens distribution → more uniform sampling across tokens

```
Example with logits [2.0, 1.0, 0.5]:

T=0.5: softmax([4.0, 2.0, 1.0]) → [0.84, 0.11, 0.05]  ← Very peaked
T=1.0: softmax([2.0, 1.0, 0.5]) → [0.59, 0.24, 0.17]  ← Original
T=2.0: softmax([1.0, 0.5, 0.25]) → [0.42, 0.31, 0.27] ← Flattened
```

---

#### Top-k Sampling

**Top-k** restricts sampling to only the k most probable tokens, preventing unlikely tokens from being selected.

```python
# Get top k tokens
top_logits, top_indices = torch.topk(logits, k=50)

# Set all other tokens to -inf
logits = torch.full_like(logits, float('-inf'))
logits.scatter_(1, top_indices, top_logits)

# Sample from filtered distribution
probs = torch.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

| Top-k Value | Effect | Trade-off |
|-------------|--------|-----------|
| **k = 1** | Greedy decoding (argmax) | No diversity |
| **k = 10-20** | Very focused | May miss good alternatives |
| **k = 40-50** | Balanced (common default) | Good diversity |
| **k = 100+** | Nearly full distribution | May include nonsense |

**Visualization:**
```
Original distribution:     Top-k=3 filtered:
Token A: 40%               Token A: 50% (40/80)
Token B: 25%          →    Token B: 31% (25/80)
Token C: 15%               Token C: 19% (15/80)
Token D: 10%               Token D: 0%  (removed)
Token E: 5%                Token E: 0%  (removed)
Token F: 5%                Token F: 0%  (removed)
```

---

#### Combining Temperature + Top-k

Best results often come from combining both techniques:

```python
def generate_with_sampling(model, idx, max_tokens, temperature=0.7, top_k=50):
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(idx[:, -context_length:])
        logits = logits[:, -1, :]
        
        # 1. Apply temperature
        logits = logits / temperature
        
        # 2. Apply top-k filtering
        if top_k is not None:
            top_logits, top_idx = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, top_idx, top_logits)
        
        # 3. Sample from filtered distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    
    return idx
```

**Recommended combinations:**

| Style | Temperature | Top-k | Result |
|-------|-------------|-------|--------|
| **Precise** | 0.3 | 10 | Focused, predictable |
| **Balanced** | 0.7 | 50 | Good default for most tasks |
| **Creative** | 1.0 | 100 | Diverse, imaginative |
| **Experimental** | 1.2 | None | Wild, unpredictable |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- pip or conda package manager
- (Optional) CUDA-capable GPU

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GPT-2.git
cd GPT-2

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.9.1 | Neural network framework |
| `tiktoken` | 0.12.0 | GPT-2 BPE tokenizer |
| `numpy` | 2.3.5 | Numerical operations |
| `matplotlib` | 3.10.8 | Loss visualization |
| `datasets` | 4.4.2 | Hugging Face data utilities |

---

## 📚 Learning Curriculum

### Recommended Study Path

```
Phase 1: Foundations
├── 1. practice/BYTEPAIRtokenizer.ipynb   → Understand tokenization
├── 2. practice/TOKENEMBEDINGS.ipynb      → Learn embedding mechanics
└── 3. practice/1.ipynb, 2.ipynb          → Core PyTorch concepts

Phase 2: Attention Mechanism
└── 4. multiheadAttention.ipynb           → Deep dive into self-attention

Phase 3: Full Architecture
└── 5. EntireGPT_architecture.ipynb       → Complete transformer block

Phase 4: Training Pipeline
└── 6. gpt-2_all_entire_pipeline.ipynb    → End-to-end training
```

---

## 📖 References

### Research Papers

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
   - *Original transformer architecture*

2. **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019)
   - [OpenAI Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
   - *GPT-2 technical report*

### Books & Resources

- **Build a Large Language Model (From Scratch)** - Sebastian Raschka
- **PyTorch Documentation** - [pytorch.org](https://pytorch.org/docs)

---

## 📊 Training Visualization

Training loss curves and metrics are documented in `loss-plot.pdf`, demonstrating the learning progression even on our minimal dataset.

---

### Technical Questions

**Q: Why use Pre-LayerNorm instead of Post-LayerNorm?**  
A: Pre-LayerNorm (normalization before attention/FFN) provides:
- Better gradient flow through deep networks
- More stable training dynamics
- Faster convergence
- Modern transformer architectures (GPT-3, BERT variants) use this approach

**Q: What is `register_buffer()` and why use it for the mask?**  
A: `register_buffer()` registers a tensor that:
- Is NOT a learnable parameter (no gradients)
- Moves with the model to GPU/CPU automatically
- Gets saved in `state_dict()` for checkpointing
- Perfect for fixed tensors like attention masks

**Q: Why does GPT-2 use GELU instead of ReLU?**  
A: GELU (Gaussian Error Linear Unit) is smoother than ReLU:
- Provides non-zero gradients for negative inputs
- Better approximates biological neuron activation
- Empirically performs better in transformer models
- Used in BERT, GPT-2, GPT-3

**Q: How does causal masking work?**  
A: Causal masking prevents the model from "cheating" by looking at future tokens:
1. Create upper triangular matrix of 1s
2. Fill masked positions with `-inf`
3. After softmax, `-inf` becomes 0 probability
4. Model can only attend to current and previous tokens

### Implementation Questions

**Q: Can I modify the model size (e.g., fewer layers, smaller embedding)?**  
A: Yes! Modify `GPT_CONFIG_124M`:
```python
GPT_CONFIG_SMALL = {
    "vocab_size": 50257,
    "context_length": 256,    # Reduced from 1024
    "emb_dim": 384,           # Reduced from 768
    "n_heads": 6,             # Reduced from 12
    "n_layers": 6,            # Reduced from 12
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

**Q: How do I save and load the trained model?**  
A:
```python
# Save
torch.save(model.state_dict(), 'gpt2_checkpoint.pth')

# Load
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load('gpt2_checkpoint.pth'))
model.eval()
```

**Q: Why is my model generating repetitive text?**  
A: Common causes:
- **Too small dataset**: Model memorizes patterns
- **Low temperature**: Use temperature > 0.7 for diversity
- **No top-k sampling**: Implement top-k or nucleus sampling
- **Insufficient training**: Model hasn't learned varied patterns

**Q: How can I visualize attention weights?**  
A: Check out `attention_visualization.ipynb` in this repository. You'll need to modify your `MultiHeadAttention` class to return attention weights.

### Training Questions

**Q: How long does training take?**  
A: On our small datasets:
- CPU: 30-60 minutes
- GPU (consumer): 5-15 minutes
- Depends on: batch size, number of epochs, hardware

**Q: What batch size should I use?**  
A: Depends on your hardware:
- **4GB RAM/VRAM**: batch_size = 4-8
- **8GB RAM/VRAM**: batch_size = 16-32
- **16GB+ RAM/VRAM**: batch_size = 64+
- Start small and increase until you hit memory limits

**Q: Should I use GPU or CPU?**  
A: 
- **GPU**: 5-10x faster, recommended if available
- **CPU**: Works fine for learning, just slower
- **MPS (Mac M1/M2)**: Supported by PyTorch 2.0+

### Troubleshooting

**Q: I'm getting "CUDA out of memory" errors**  
A: Solutions:
1. Reduce batch size
2. Reduce context length
3. Use gradient accumulation
4. Enable mixed precision training (FP16)

**Q: Loss is not decreasing**  
A: Check:
- Learning rate (try 3e-4 to 1e-3)
- Data preprocessing (tokens in correct range?)
- Model initialization (weights properly initialized?)
- Gradient clipping (prevent exploding gradients)

**Q: How do I know if my model is learning?**  
A: Signs of learning:
- Training loss decreases over time
- Validation loss decreases (if you have val set)
- Generated text becomes more coherent
- Perplexity decreases

### Advanced Topics

**Q: Can I fine-tune this on my own dataset?**  
A: Yes! Steps:
1. Prepare your text data
2. Tokenize using tiktoken
3. Create DataLoader
4. Load pre-trained weights (if available)
5. Train with lower learning rate (1e-5 to 1e-4)

**Q: How do I implement beam search?**  
A: Beam search is more complex than greedy/sampling. Key steps:
1. Maintain top-k hypotheses at each step
2. Expand each hypothesis with top-k next tokens
3. Keep top-k overall sequences
4. Return highest probability sequence

**Q: Can I use this with other tokenizers?**  
A: Yes, but you'll need to:
1. Update `vocab_size` in config
2. Modify tokenization code
3. Ensure special tokens are handled correctly
4. Retrain embeddings from scratch

---

## 🤝 Contributing

Contributions to improve explanations, add visualizations, or fix bugs are welcome. Please feel free to open issues or submit pull requests.

---

## 📄 License

This project is intended for educational purposes. Feel free to use, modify, and distribute for learning.

---

<div align="center">

### 💡 "The best way to understand deep learning is to implement it from scratch."

<br>

**Built for Learning | Implemented with PyTorch | Inspired by OpenAI's GPT-2**

---

*If this repository helped you understand transformers better, consider giving it a ⭐*

</div>
