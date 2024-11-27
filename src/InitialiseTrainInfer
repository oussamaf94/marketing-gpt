model = GPT(config).to("cuda")

# Train the model
train_model(model, segments_tensor, messages_tensor, config)

def generate_marketing_messagez(model, segment_text, max_length=50, temperature=0.7, top_k=50):
    """
    Generate a marketing message using the trained GPT model.

    Args:
        model: Trained GPT model
        segment_text: String containing the customer segment description
        max_length: Maximum length of generated message
        temperature: Controls randomness (0.7 = balanced, <0.7 = focused, >0.7 = creative)
        top_k: Number of top tokens to consider for sampling
    """
    # Tokenize the segment
    #model.eval()
    segment_tokens = encode(segment_text, config['seg_length'])
    segment_tensor = torch.tensor(segment_tokens).unsqueeze(0).to('cuda')

    # Generate message tokens
    generated_tokens = model.generate(
        segment=segment_tensor,
        max_new_tokens=max_length,
        temperature=temperature,
        top_k=top_k
    )

    # Decode the generated tolist())

    return message

segment = "Young urban man"
message = generate_marketing_messagez(
    model=model,
    segment_text=segment,
    max_length=64,
    temperature=0.7,
    top_k=50
    )

print(f"Segment: {segment}")
print(f"Generated Message: {message}")
