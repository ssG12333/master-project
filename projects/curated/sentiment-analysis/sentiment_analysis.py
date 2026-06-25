import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import os
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm

# 下载 NLTK 必要资源
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 显存占用显示函数
def print_memory_usage(device):
    if torch.cuda.is_available() and device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(device) / 1024**2   # MB


# 1. 数据处理
class CustomIMDBDataset(Dataset):
    def __init__(self, reviews, labels, vocab, max_len=300):
        self.reviews = reviews
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])  # 确保输入为字符串
        tokens = word_tokenize(review.lower())
        indices = [self.vocab.get(token, 0) for token in tokens]  # 0 为未知词
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

# 加载和预处理数据
def load_csv_data(csv_path, max_words=10000, max_len=300):
    print("Loading CSV file...")
    df = pd.read_csv(csv_path)
    reviews = df['review'].tolist()
    labels = [1 if sentiment == 'positive' else 0 for sentiment in df['sentiment']]

    print("Building vocabulary...")
    all_tokens = []
    for review in tqdm(reviews, desc="Tokenizing reviews"):
        all_tokens += word_tokenize(str(review).lower())
    token_counts = Counter(all_tokens)
    vocab = {token: idx + 1 for idx, (token, _) in enumerate(token_counts.most_common(max_words - 1))}
    vocab['<PAD>'] = 0

    print("\nSplitting dataset...")
    train_size = int(0.8 * len(reviews))
    train_reviews, val_reviews = reviews[:train_size], reviews[train_size:]
    train_labels, val_labels = labels[:train_size], labels[train_size:]

    train_dataset = CustomIMDBDataset(train_reviews, train_labels, vocab, max_len)
    val_dataset = CustomIMDBDataset(val_reviews, val_labels, vocab, max_len)

    return train_dataset, val_dataset, vocab

# 2. BiLSTM 模型定义
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                              bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.bilstm(embedded)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out.squeeze())

# 3. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_path):
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
            print_memory_usage(device)

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loop.set_postfix(loss=loss.item())
                print_memory_usage(device)

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct / total)

        # 动态调整学习率
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%')

    # 保存损失和准确率数据
    with open(save_path, 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'val_losses': val_losses,
                     'val_accuracies': val_accuracies}, f)

    return train_losses, val_losses, val_accuracies

# 4. 绘制损失和准确率曲线
def plot_metrics(save_path, title):
    with open(save_path, 'rb') as f:
        metrics = pickle.load(f)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_losses'], label='Train Loss')
    plt.plot(metrics['val_losses'], label='Validation Loss')
    plt.title(f'{title} - Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics['val_accuracies'], label='Validation Accuracy')
    plt.title(f'{title} - Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

# 5. 主实验函数
def run_experiment(csv_path, learning_rate, num_layers, batch_size, save_path, num_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print_memory_usage(device)

    # 数据加载
    train_dataset, val_dataset, vocab = load_csv_data(csv_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 模型参数
    vocab_size = len(vocab) + 1  # 包括填充词
    embed_dim = 256
    hidden_dim = 512

    # 初始化模型
    model = BiLSTMClassifier(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 训练
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_path)

    # 测试（使用验证集作为测试集）
    model.eval()
    correct = 0
    total = 0
    test_loop = tqdm(val_loader, desc='Testing', leave=False)
    with torch.no_grad():
        for inputs, labels in test_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print_memory_usage(device)

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # 绘制结果
    plot_metrics(save_path, f'BiLSTM lr={learning_rate} layers={num_layers} batch={batch_size}')

    return test_accuracy

# 6. 消融实验
def main():
    csv_path = 'IMDB Dataset.csv'  # 替换为你的 CSV 文件路径
    learning_rates = [0.005, 0.0005]
    num_layers_list = [1, 2]
    batch_sizes = [64, 128]
    results = {}

    for lr in learning_rates:
        for layers in num_layers_list:
            for batch_size in batch_sizes:
                save_path = f'metrics_lr{lr}_layers{layers}_batch{batch_size}.pkl'
                print(f'\nRunning experiment with lr={lr}, layers={layers}, batch_size={batch_size}')
                test_acc = run_experiment(csv_path, lr, layers, batch_size, save_path)
                results[(lr, layers, batch_size)] = test_acc

    # 打印结果总结
    print("\nExperiment Results Summary:")
    for (lr, layers, batch_size), acc in results.items():
        print(f'Learning Rate: {lr}, Layers: {layers}, Batch Size: {batch_size}, Test Accuracy: {acc:.2f}%')

if __name__ == '__main__':
    main()