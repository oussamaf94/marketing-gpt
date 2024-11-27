class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["vocab_size"] is not None
        assert config["block_size"] is not None
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config["vocab_size"], config["n_embed"]),
            'wpe': nn.Embedding(config["block_size"], config["n_embed"]),
            'drop': nn.Dropout(config["dropout"]),
            'h': nn.ModuleList([Block(config) for _ in range(config["n_layers"])]),
            'ln_f': nn.LayerNorm(config["n_embed"])
        })

        self.lm_head = nn.Linear(config["n_embed"], config["vocab_size"], bias=False)

        # Tie weights between embedding and output layer
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize embedding layers
        nn.init.normal_(self.transformer.wte.weight, std=0.02)
        nn.init.normal_(self.transformer.wpe.weight, std=0.02)

        # Initialize layer norm
        nn.init.normal_(self.transformer.ln_f.weight, std=0.02)
        nn.init.zeros_(self.transformer.ln_f.bias)

    def forward(self, segment, message=None):
        device = segment.device

        # Combine segment and message
        if message is not None:

            idx = torch.cat([segment, message], dim=1)
        
        else:
            
            idx = segment


        B, T = idx.size()
        # Block size check
        assert T <= self.config["block_size"], f"Cannot forward sequence of length {T}, block size is {self.config['block_size']}"

        # Create position tensor directly on correct device
        pos = torch.arange(0, T, device=device).unsqueeze(0)

        # Embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm and projection
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Loss computation
        loss = None
        if message is not None:
            # Get message logits and compute loss
            message_start = segment.size(1)
            logits_message = logits[:, message_start:, :]

            loss = F.cross_entropy(
                logits_message.reshape(-1, logits.size(-1)),
                message.reshape(-1),
                ignore_index=-1
            )

        return logits, loss

    def generate(self, segment, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()  # Set to evaluation mode

        with torch.no_grad():
            seg_len = segment.size(1)  # Store original segment length
            for _ in range(max_new_tokens):
                if segment.size(1) > self.config["block_size"]:
                    idx_cond = segment[:, -self.config["block_size"]:]
                else:
                    idx_cond = segment

                # Get predictions
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature  # Take the logits for the last token

                # Apply top-k sampling if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Concatenate to segment
                segment = torch.cat((segment, next_token), dim=1)

                # Optional: Stop if end token is generated
                if next_token.item() == self.config.get("eos_token", -1):
                    break

            # Return generated sequence excluding input segment
            return segment[:, seg_len:]
