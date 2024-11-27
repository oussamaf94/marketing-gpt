import torch
import torch.nn as nn
import torch.optim as optim

chars = sorted(set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!-()\''))
vocab_size = len(chars) + 1

config = {
    "vocab_size":vocab_size,
    "n_embed": 512,         
    "n_heads": 8,          
    "seg_length": 44,        
    "message_length": 70,   
    "dropout": 0.1,
    "block_size": 44 + 70,
    "n_layers": 5,
    "batch_size": 128,
    "n_epochs": 10
}


segments = []
messages = []

# Extract and convert each segment and message to a tensor
for segment_encoded, message_encoded in encoded_pairs:
    segments.append(torch.tensor(segment_encoded))  # Convert to tensor
    messages.append(torch.tensor(message_encoded))  # Convert to tensor

# Stack the padded segments and messages into tensors (batch first)
segments_tensor = torch.stack(segments)  # Shape: (batch_size, seg_length)
messages_tensor = torch.stack(messages)  # Shape: (batch_size, msg_length)

batch_size = 32  # Number of rows per batch
total_samples = segments_tensor.size(0)  # Number of rows in the dataset
