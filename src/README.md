# Source Code Documentation

## Model Architecture Overview

The model follows a hierarchical structure where each component builds upon the previous one:

1. **CausalSelfAttention**
   - Implements custom attention mechanism with segment-aware masking
   - Handles both segment and message portions differently
   - Mask is dynamically sliced during inference, but fixed during training to allow for batch optimisation
   - Uses scaled dot-product attention with parallel heads
   - Includes dropout and residual connections

2. **MLP (Multi-Layer Perceptron)**
   - Processes transformed attention outputs
   - Uses GELU activation function
   - Includes dropout for regularization
   - Projects to and from larger intermediate dimensions

3. **Block**
   - Combines CausalSelfAttention and MLP
   - Implements layer normalization
   - Handles residual connections
   - Forms the basic transformer block unit

4. **GPT**
   - Main model class that ties everything together
   - Manages token and position embeddings
   - Stacks multiple Blocks
   - Handles generation logic and loss computation

## Module Relation
GPT
└── Block
        └── MLP
              └──CausalSelfAttention

