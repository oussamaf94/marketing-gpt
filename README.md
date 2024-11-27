# Marketing Message GPT

A transformer-based GPT model for generating targeted marketing messages, implemented
following the architecture from "Attention Is All You Need" and inspired by 
Andrej Karpathy's minGPT youtube series.

## Implementation Details

- Based on the transformer architecture (Vaswani et al., 2017)
- Custom causal attention mechanism for segment-aware generation
- Efficient implementation following Karpathy's minGPT principles

## Important Note

This model has been trained only on synthetic, generated data and is not suitable for production use. The training data was artificially created for demonstration purposes, and the model's outputs should not be considered representative of real marketing messages.


## Model Configuration

The model uses the following default configuration:
```python
config = {
    "n_layers": 8,
    "n_heads": 8,
    "n_embed": 512,
    "block_size": 114,
    "dropout": 0.1,
    "seg_length": 44,
    "message_length": 70
}

## Quick Start

```python
from src.model import GPT
from src.utils import generate_marketing_message

# Initialise model
model = GPT(config)

# Generate message
segment = "young urban professionals interested in fitness"
message = generate_marketing_message(
    model,
    segment,
    temperature=0.7,
    top_k=50
)
