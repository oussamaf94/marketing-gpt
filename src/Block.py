class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ln_1 = nn.LayerNorm(config["n_embed"])
        self.ln_2 = nn.LayerNorm(config["n_embed"])
        self.ff = MLP(config)  # Using the updated MLP class name

        # Optional: Initialize layer norm parameters
        self._init_weights()

    def _init_weights(self):
        # Initialize layer norm weights and biases
        self.ln_1.bias.data.zero_()
        self.ln_1.weight.data.fill_(1.0)
        self.ln_2.bias.data.zero_()
        self.ln_2.weight.data.fill_(1.0)

    def forward(self, x):
        # Residual connections maintain device of input tensor
        x = x + self.attn(self.ln_1(x))  # First residual connection
        x = x + self.ff(self.ln_2(x))    # Second residual connection
        return x
