# Define vocabulary (alphabet, numbers, spaces, and punctuation)
# Define character set and vocabulary
chars = sorted(set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!-()\''))
stoi = {ch:i+1 for i,ch in enumerate(chars)}  # Shift all indices by 1
itos = {i+1:ch for i,ch in enumerate(chars)}  # Shift all indices by 1


def encode(sentence, length=seg_max):
    indicies = []
    for c in sentence:
        indicies.append(stoi[c])

    # Add padding

    if length is not None and len(indicies) < length:
        indicies += [0] * (length - len(indicies))

    return indicies

def decode(sequence):
    sentence = []
    for i in sequence:
        if i != 0:  # Skip padding tokens
            sentence.append(itos[i])
    return ''.join(sentence)
