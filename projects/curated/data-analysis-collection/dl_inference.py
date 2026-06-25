import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools  # 用于生成超参数组合

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. 数据加载与预处理 ---
# 加载CSV文件
# 假设 'combined_fiberMS_1_5_5021data.csv' 文件与脚本在同一目录下
try:
    df = pd.read_csv('combined_fiberMS_1_5_5021data.csv')
    print("数据加载成功。")
except FileNotFoundError:
    print("错误：'combined_fiberMS_1_5_5021data.csv' 文件未找到。请确保文件在正确目录或可访问。")
    # 如果文件未找到，程序将退出
    exit()

# --- 检查并打印实际列名 ---
print("\nCSV文件中实际的列名：")
print(df.columns.tolist())
print("\n请对照上述列名，确保代码中使用的特征名称完全匹配。")

# --- 数据筛选：移除 'number of hits' 为 0 的数据行 ---
if 'number of hits' in df.columns:
    initial_rows = len(df)
    df = df[df['number of hits'] > 0].copy()  # 使用 .copy() 避免SettingWithCopyWarning
    filtered_rows = len(df)
    print(f"\n已移除 {initial_rows - filtered_rows} 行 'number of hits' 为 0 的数据。")
    print(f"剩余 {filtered_rows} 行数据用于模型训练。")
else:
    print("警告：'number of hits' 列不存在。无法进行数据筛选。")
    exit("关键列 'number of hits' 不存在，程序无法继续。请检查您的CSV文件。")

# 定义输入 (X) 和输出 (y) 特征
# 输入特征是物理模型的输出值
# 输出特征是我们要反演的物理模型输入变量
# 动态调整输入和输出特征列表，只包含数据中实际存在的列
original_input_features_for_dl = ['number of hits', 'first hit time', 'first hit location', 'DSS', 'DAS']
original_output_features_for_dl = ['Orientation', 'Density', 'Length']

# 检查实际存在的列，并更新特征列表
input_features_for_dl = [col for col in original_input_features_for_dl if col in df.columns]
output_features_for_dl = [col for col in original_output_features_for_dl if col in df.columns]

missing_input_cols = [col for col in original_input_features_for_dl if col not in df.columns]
missing_output_cols = [col for col in original_output_features_for_dl if col not in df.columns]

if missing_input_cols:
    print(f"警告：以下输入列在数据中缺失，将不会用于模型训练：{missing_input_cols}")
if missing_output_cols:
    print(f"错误：以下输出列在数据中缺失，程序无法继续：{missing_output_cols}")
    exit()

if not input_features_for_dl or not output_features_for_dl:
    print("错误：没有足够的输入或输出列来构建模型。请检查您的CSV文件和列名定义。")
    exit()

X = df[input_features_for_dl]
y = df[output_features_for_dl]

# 对输入特征进行Min-Max标准化
# Min-Max标准化将特征缩放到 [0, 1] 范围，这对于神经网络的训练非常有利，
# 尤其是在数据分布不明确（非正态）且存在非线性关系时。
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# 划分数据集：训练集、验证集和测试集
# 训练集用于模型学习，验证集用于调整超参数和监控过拟合，测试集用于最终评估模型性能。
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# 将临时集合 (X_temp, y_temp) 划分为验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\n数据集划分完成：")
print(f"训练集 X_train 形状: {X_train.shape}, y_train 形状: {y_train.shape}")
print(f"验证集 X_val 形状: {X_val.shape}, y_val 形状: {y_val.shape}")
print(f"测试集 X_test 形状: {X_test.shape}, y_test 形状: {y_test.shape}")

# 将 numpy 数组转换为 PyTorch 张量
# 确保数据类型为 float32 用于神经网络计算
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)  # .values 获取 DataFrame 的 numpy 数组
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


# --- 2. PyTorch Dataset and DataLoader ---

# 自定义 PyTorch Dataset 类
class FractureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建 Dataset 实例
train_dataset = FractureDataset(X_train_tensor, y_train_tensor)
val_dataset = FractureDataset(X_val_tensor, y_val_tensor)
test_dataset = FractureDataset(X_test_tensor, y_test_tensor)


# --- 3. PyTorch 模型定义 (TransformerRegressor) ---

# 定义基于 Transformer 编码器的回归模型架构
# 将输入特征投影到 d_model 维度，然后通过 Transformer 编码器。
# Transformer 编码器由多个 TransformerEncoderLayer 组成，每个层包含多头自注意力机制和前馈网络。
# Dropout 层用于防止过拟合。
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout_rate):
        super(TransformerRegressor, self).__init__()
        # 将输入特征投影到 Transformer 的 d_model 维度
        self.input_projection = nn.Linear(input_dim, d_model)

        # 定义 Transformer 编码器层
        # batch_first=True 表示输入张量的形状是 (batch_size, sequence_length, feature_dimension)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                                   batch_first=True)
        # 定义 Transformer 编码器，由多个编码器层堆叠而成
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 输出层，将 Transformer 的输出映射到回归目标
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # 将输入特征投影到 d_model 维度
        x = self.input_projection(x)  # shape: (batch_size, d_model)

        # Transformer 编码器期望的输入形状是 (batch_size, sequence_length, feature_dimension)。
        # 对于表格数据，我们将整个特征向量视为一个序列长度为 1 的“token”。
        # 添加一个序列长度维度: (batch_size, 1, d_model)
        x = x.unsqueeze(1)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)  # shape: (batch_size, 1, d_model)

        # 移除序列长度维度，并通过输出层
        x = x.squeeze(1)  # shape: (batch_size, d_model)
        output = self.output_layer(x)
        return output


input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

# --- 4. 网格调参设置 ---
# 定义超参数网格
# 您可以根据需求和计算资源添加更多选项或不同的超参数
param_grid = {
    'learning_rate': [0.001, 0.0005],
    'batch_size': [64, 128],
    'd_model': [64, 128],  # Transformer 模型的特征维度
    'nhead': [4, 8],  # 多头注意力机制的头数
    'num_encoder_layers': [2, 3],  # Transformer 编码器层的数量
    'dim_feedforward': [128, 256],  # 前馈网络模型的维度
    'dropout_rate': [0.1, 0.2]  # Dropout 率
}

best_val_loss = float('inf')
best_params = None
best_model_state = None
best_history = None

all_param_combinations = list(itertools.product(*param_grid.values()))
print(f"\n总共需要训练 {len(all_param_combinations)} 个模型。")

# 确定设备 (CPU 或 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n使用设备: {device}")

# 遍历每个超参数组合
for i, combo in enumerate(all_param_combinations):
    current_params = dict(zip(param_grid.keys(), combo))
    print(f"\n--- 训练模型 {i + 1}/{len(all_param_combinations)} ---")
    print(f"当前参数: {current_params}")

    # 为当前 batch_size 创建 DataLoader 实例
    train_loader = DataLoader(train_dataset, batch_size=current_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=current_params['batch_size'], shuffle=False)

    # 使用当前超参数初始化模型
    model = TransformerRegressor(input_dim, output_dim,
                                 d_model=current_params['d_model'],
                                 nhead=current_params['nhead'],
                                 num_encoder_layers=current_params['num_encoder_layers'],
                                 dim_feedforward=current_params['dim_feedforward'],
                                 dropout_rate=current_params['dropout_rate']).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=current_params['learning_rate'])

    num_epochs = 200
    patience = 20
    current_min_val_loss = float('inf')
    epochs_no_improve = 0

    current_history = {'loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        current_history['loss'].append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        current_history['val_loss'].append(val_loss)

        # print(f"  Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}") # 调试时可取消注释

        # 早停逻辑
        if val_loss < current_min_val_loss:
            current_min_val_loss = val_loss
            epochs_no_improve = 0
            # 暂时保存此组合的最佳状态字典
            temp_best_state_dict = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                # print(f"  早停触发，在 {epoch+1} 轮次后停止。") # 调试时可取消注释
                break

    print(f"  当前组合最佳验证损失: {current_min_val_loss:.6f}")

    # 检查此组合是否是整体最佳
    if current_min_val_loss < best_val_loss:
        best_val_loss = current_min_val_loss
        best_params = current_params
        best_model_state = temp_best_state_dict  # 保存最佳模型状态
        best_history = current_history  # 保存最佳模型的训练历史

print("\n--- 网格调参完成 ---")
print(f"最佳超参数组合: {best_params}")
print(f"最佳验证损失: {best_val_loss:.6f}")

# --- 5. 使用最佳模型进行最终评估 ---
print("\n--- 使用最佳模型进行最终评估 ---")
# 使用最佳参数重新初始化模型并加载其状态
final_model = TransformerRegressor(input_dim, output_dim,
                                   d_model=best_params['d_model'],
                                   nhead=best_params['nhead'],
                                   num_encoder_layers=best_params['num_encoder_layers'],
                                   dim_feedforward=best_params['dim_feedforward'],
                                   dropout_rate=best_params['dropout_rate']).to(device)
final_model.load_state_dict(best_model_state)
final_model.eval()  # 设置为评估模式

# 使用最佳批次大小为测试集创建 DataLoader
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

# 在测试集上进行预测
y_pred_tensor = []
y_test_actual_tensor = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = final_model(inputs)
        y_pred_tensor.append(outputs.cpu())  # 移动到 CPU 以便进行 numpy 转换
        y_test_actual_tensor.append(targets.cpu())

y_pred = torch.cat(y_pred_tensor).numpy()
y_test_actual = torch.cat(y_test_actual_tensor).numpy()

# 计算每个输出变量的指标
results = {}
for i, output_name in enumerate(output_features_for_dl):
    actual = y_test_actual[:, i]
    predicted = y_pred[:, i]

    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    results[output_name] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}
    print(f"\n{output_name} 的评估指标 (测试集):")
    print(f"  均方误差 (MSE): {mse:.6f}")
    print(f"  均方根误差 (RMSE): {rmse:.6f}")
    print(f"  R平方 (R2): {r2:.6f}")


# --- 6. 结果可视化 (使用最佳模型的训练历史) ---

# 创建预测值 vs 真实值图的函数
def plot_predicted_vs_actual(y_true, y_pred, feature_name, metrics):
    plt.figure(figsize=(8, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, color='steelblue')

    # 添加完美拟合线 (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美拟合线')

    plt.title(f'预测值 vs 真实值 ({feature_name})')
    plt.xlabel(f'真实 {feature_name} 值')
    plt.ylabel(f'预测 {feature_name} 值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 在图上显示指标
    text_str = f"MSE: {metrics['MSE']:.6f}\nRMSE: {metrics['RMSE']:.6f}\nR2: {metrics['R2']:.6f}"
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    plt.tight_layout()
    plt.show()


# 使用最佳模型的预测结果为每个输出特征绘制图表
for i, output_name in enumerate(output_features_for_dl):
    plot_predicted_vs_actual(y_test_actual[:, i], y_pred[:, i], output_name, results[output_name])

# 绘制最佳模型的训练损失和验证损失曲线
if best_history:
    plt.figure(figsize=(10, 6))
    plt.plot(best_history['loss'], label='训练损失')
    plt.plot(best_history['val_loss'], label='验证损失')
    plt.title(f'最佳模型损失随训练轮次变化 (最佳参数: {best_params})')
    plt.xlabel('训练轮次 (Epoch)')
    plt.ylabel('损失 (MSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 清理保存的模型文件 (如果在此过程中有保存，尽管网格搜索不严格需要)
# 最佳模型状态保存在内存中 (best_model_state)
# 如果您想将最终的最佳模型保存到磁盘，可以添加：
# torch.save(best_model_state, 'final_best_model_pytorch.pth')
# 然后如果需要可以删除它：
# if os.path.exists('final_best_model_pytorch.pth'):
#     os.remove('final_best_model_pytorch.pth')
