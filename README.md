# Marketing Message GPT

A transformer-based GPT model for generating targeted marketing messages, implemented
following the architecture from "Attention Is All You Need" and inspired by 
Andrej Karpathy's minGPT youtube series.

## Implementation Details

- Based on the transformer architecture (Vaswani et al., 2017)
- Custom causal attention mechanism for segment-aware generation
- Efficient implementation following Karpathy's minGPT principles

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
