import torch

def generate_text(model, tokenizer, start_text, max_length=50):
    model.eval()
    generated = tokenizer.encode(start_text)
    input_seq = torch.tensor(generated, dtype=torch.long).unsqueeze(0)
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_seq)
            next_token = torch.argmax(outputs[:, -1, :], dim=-1).item()
            generated.append(next_token)
            input_seq = torch.tensor(generated[-len(input_seq[0]):], dtype=torch.long).unsqueeze(0)
    
    return tokenizer.decode(generated)
