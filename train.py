import torch
from torch.utils.data import DataLoader
from data.dataset import TextDataset
from models.gpt_model import MyGPT
from utils.tokenizer import SimpleTokenizer
import torch.optim as optim
import torch.nn as nn
from config import load_config
import pickle

# load config
config = load_config()
embed_dim = config['embed_dim']
num_heads = config['num_heads']
num_layers = config['num_layers']
seq_length = config['seq_length']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
epochs = config['epochs']

# load data
text = "Artificial intelligence (AI) has become an integral part of modern technology and everyday life. From voice assistants like Siri and Alexa to advanced algorithms that power search engines and social media platforms, AI is revolutionizing the way we interact with the world. Machine learning, a subset of AI, enables computers to learn from data and make decisions or predictions without being explicitly programmed for every possible scenario. This technology is used in various industries, including healthcare, finance, transportation, and entertainment. In healthcare, AI can analyze medical images to detect diseases like cancer at an early stage, often with higher accuracy than human doctors. In finance, AI algorithms are used for fraud detection, stock trading, and customer service chatbots. Self-driving cars are another prominent application of AI in transportation, where sensors and machine learning models work together to navigate roads and avoid obstacles. As AI continues to evolve, ethical concerns about data privacy, bias, and job displacement arise. It is crucial to address these challenges to ensure AI benefits society as a whole. Researchers are working to create fair and transparent AI systems that minimize bias and operate within ethical guidelines. As technology advances, the future of AI holds immense potential, offering solutions to complex problems and improving the quality of life for people worldwide."
tokenizer = SimpleTokenizer(text)
vocab_size = len(tokenizer.vocab)
dataset = TextDataset(text, tokenizer.vocab, seq_length)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
model = MyGPT(vocab_size, embed_dim, num_heads, num_layers, seq_length)

# 定义损失函数和优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(epochs):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        
        outputs = outputs.view(-1, vocab_size)
        targets = targets.view(-1)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 保存 tokenizer
with open("./model_files/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

torch.save(model.state_dict(), "model_files/gpt_model.pth")

print("模型和 tokenizer 已保存至model_files文件夹中")