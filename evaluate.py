import torch
from models.gpt_model import MyGPT  # 导入 GPT 模型
from config import load_config
from generate import generate_text
import pickle

# 加载 tokenizer 并动态获取 vocab_size
def load_tokenizer(tokenizer_path):
    # 加载 tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.vocab)  # 动态获取 vocab_size
    print(f"Tokenizer 已加载, vocab_size: {vocab_size}")
    return tokenizer, vocab_size

# 加载模型
def load_model(config_path, model_path, vocab_size):
    # 加载配置
    config = load_config(config_path)
    
    # 创建 GPT 模型，使用动态获取的 vocab_size
    model = MyGPT(vocab_size=vocab_size,
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                seq_length=config['seq_length'])
    
    # 加载模型参数，设置 weights_only=True
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # 切换到评估模式
    model.eval()
    print("模型已成功加载并切换到评估模式")
    
    return model


# 主评估函数
if __name__ == "__main__":
    # 配置文件路径和模型路径
    config_path = 'config.yaml'
    model_path = 'model_files/gpt_model.pth'
    tokenizer_path = "model_files/tokenizer.pkl"
    
    # 加载 tokenizer 并动态获取 vocab_size
    tokenizer, vocab_size = load_tokenizer(tokenizer_path)
    
    # 加载模型，使用动态获取的 vocab_size
    model = load_model(config_path, model_path, vocab_size)
    
    # 输入起始文本
    start_text = "I like ai"
    
    # 生成文本
    generated_text = generate_text(model, tokenizer, start_text)
    print("生成的文本:", generated_text)
