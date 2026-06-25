import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义CNN模型
class StockCNN(nn.Module):
    def __init__(self,dropout_rate=0.5):
        super(StockCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, kernel_size=(1,19))
        self.conv2 = nn.Conv2d(7, 7, kernel_size=(3,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,1))
        self.conv3 = nn.Conv2d(7, 7, kernel_size=(3,1))
        self.pool2=nn.MaxPool2d(kernel_size=(2,1))
        self.fc=nn.Linear(77,1)
        self.dropout=nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加一个channel维度，尺寸变为 (B, 1, 50, 19)
        x = self.conv1(x)  # 输出尺寸 (B, 7, 50, 1)
        x = nn.ReLU()(x)

        x = self.conv2(x)  # 输出尺寸 (B, 7, 48, 1)
        x = nn.ReLU()(x)
        x = self.pool1(x)  # 输出尺寸 (B, 7, 24, 1)

        x = self.conv3(x)  # 输出尺寸 (B, 7, 22, 1)
        x = nn.ReLU()(x)
        x = self.pool2(x)  # 输出尺寸 (B, 7, 11, 1)

        x = x.view(-1, 77)  # 展平操作，输出尺寸 (B, 77)
        x = self.dropout(x)
        x = self.fc(x)  # 全连接层，输出尺寸 (B, 1)
        x = self.sigmoid(x)  # Sigmoid激活函数，输出尺寸 (B, 1)
        return x

# 创建Patching函数
def create_patches(data, window_size, label_column='label'):
    X, y = [], []
    features = data.columns.difference(['date', label_column])
    for i in range(len(data) - window_size):
        patch = data.iloc[i:i + window_size][features].values
        label = data.iloc[i + window_size][label_column]

        # 对每个Patch进行标准化
        patch = (patch - patch.mean(axis=0)) / patch.std(axis=0)

        X.append(patch)
        y.append(label)
    return np.array(X), np.array(y)

# 处理单个数据集的函数
def process_dataset(file_path, output_dir, window_size=50, num_epochs=100, batch_size=32):
    # 读取数据
    data = pd.read_csv(file_path)
    # 确保日期列是datetime类型
    data['date'] = pd.to_datetime(data['date'])
    data = data.fillna(method='ffill').fillna(method='bfill')

    # 生成标签，假设 'close' 是收盘价
    data['label'] = (data['close'].shift(-1) > data['close']).astype(int)
    # 丢弃最后一行，因为它没有标签
    data = data[:-1]

    # 创建数据补丁
    X, y = create_patches(data, window_size)

    # 因为Patching之后数据长度变短了，所以我们需要调整划分的时间点
    split_indices = [
        len(data[(data['date'] >= '2020-01-01') & (data['date'] <= '2023-03-31')]) - window_size,
        len(data[(data['date'] >= '2020-01-01') & (data['date'] <= '2024-05-31')]) - window_size
    ]

    X_train, y_train = X[:split_indices[0]], y[:split_indices[0]]
    X_val, y_val = X[split_indices[0]:split_indices[1]], y[split_indices[0]:split_indices[1]]
    X_test, y_test = X[split_indices[1]:], y[split_indices[1]:]

    # 转换为Pytorch的Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # 创建数据加载器
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # 实例化模型
    model = StockCNN().to(device)
    # 损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

    # 绘制损失曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 测试模型并计算AUC、ACC、MCC
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            test_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 计算AUC、ACC、MCC
    auc = roc_auc_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, (all_preds > 0.7).astype(int))
    mcc = matthews_corrcoef(all_targets, (all_preds > 0.7).astype(int))

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"MCC: {mcc:.4f}")

    # 创建保存指标的DataFrame
    metrics_df = pd.DataFrame({
        'AUC': [auc],
        'Accuracy': [acc],
        'MCC': [mcc]
    }, index=[os.path.basename(file_path).split('_')[0]])

    # 保存指标到CSV文件
    metrics_path = os.path.join(output_dir, 'metrics2.csv')
    if not os.path.exists(metrics_path):
        metrics_df.to_csv(metrics_path)
    else:
        metrics_df.to_csv(metrics_path, mode='a', header=False)

    # 创建保存预测值的DataFrame
    preds_df = pd.DataFrame(all_preds, columns=['Prediction'])
    preds_df['Date'] = data.iloc[split_indices[1]+50:]['date'].values
    preds_df.set_index('Date', inplace=True)

    # 保存预测值到CSV文件
    preds_path = os.path.join(output_dir, f'predictions_{os.path.basename(file_path).split("_")[0]}.csv')
    preds_df.to_csv(preds_path)

# 设置数据文件夹和输出
data_folder = 'datav2'
output_folder = 'Data3'

# 指定要处理的股票代码
stock_code = '600150'  # 修改为你想处理的股票代码
file_name = f'{stock_code}_processed.csv'
file_path = os.path.join(data_folder, file_name)

if os.path.isfile(file_path):
    print(f"Processing file: {file_name}")
    process_dataset(file_path, output_folder)
else:
    print(f"File not found: {file_path}")
