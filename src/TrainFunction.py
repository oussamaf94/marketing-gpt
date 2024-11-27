def train_model(model, segments_tensor, messages_tensor, config):
    device = next(model.parameters()).device
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["n_epochs"])

    # Calculate total batches
    total_samples = len(segments_tensor)
    batch_size = config["batch_size"]
    n_batches = (total_samples + batch_size - 1) // batch_size

    model.train()
    for epoch in range(config["n_epochs"]):
        total_loss = 0

        for i in range(0, total_samples, batch_size):
            # Get batch
            batch_segments = segments_tensor[i:i+batch_size]
            batch_messages = messages_tensor[i:i+batch_size]

            # Move to GPU
            segment = batch_segments.to(device, non_blocking=True)
            message = batch_messages.to(device, non_blocking=True)

            # Forward pass
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            logits, loss = model(segment, message)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

            # Print progress
            batch_idx = i // batch_size
            if batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{config['n_epochs']}, "
                      f"Batch {batch_idx}/{n_batches}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Avg Loss: {avg_loss:.4f}")

        # Step the learning rate scheduler
        scheduler.step()

        # Calculate epoch statistics
        epoch_loss = total_loss / n_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss:.4f}")
