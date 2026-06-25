import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datetime
import os
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')

# 设置Seaborn风格
sns.set_style("whitegrid")
sns.set_palette("muted")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建统一输出文件夹
output_dir = '输出结果'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)
# 自定义Transformer编码器层
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.2):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)  # 2层
    def forward(self, x):
        return self.transformer_encoder(x)
# MLP+LSTM+Transformer模型
class MLPLSTMTransformerModel(nn.Module):
    def __init__(self, input_dim, mlp_hidden_dims=[256, 128], lstm_hidden_dim=128, d_model=128, nhead=8, dim_feedforward=256, dropout=0.2):
        super(MLPLSTMTransformerModel, self).__init__()
        # MLP层
        mlp_layers = []
        prev_dim = input_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)  # MLP专用Dropout
            ])
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        # LSTM层
        self.lstm = nn.LSTM(prev_dim, lstm_hidden_dim, batch_first=True, dropout=0.2 if dropout else 0)
        # Transformer层
        self.transformer = TransformerEncoder(d_model=lstm_hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        # BatchNorm
        self.bn = nn.BatchNorm1d(lstm_hidden_dim)
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        # MLP处理输入
        batch_size, seq_len, feat_dim = x.size()
        x = x.view(-1, feat_dim)  # 展平为 (batch_size*seq_len, feat_dim)
        x = self.mlp(x)
        x = x.view(batch_size, seq_len, -1)  # 恢复形状 (batch_size, seq_len, mlp_output_dim)
        # LSTM处理序列
        lstm_out, _ = self.lstm(x)
        # BatchNorm
        lstm_out = self.bn(lstm_out.permute(0, 2, 1)).permute(0, 2, 1)
        # Transformer处理
        transformer_out = self.transformer(lstm_out)
        # 输出层
        out = self.fc(transformer_out[:, -1, :])  # 取最后一个时间步
        return out
# 早停机制
class EarlyStopping:
    def __init__(self, patience=20, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0


# 步骤1：加载和预处理CSV数据
print("步骤1：加载和预处理CSV数据...")
try:
    data = pd.read_csv('附件1_修改版.csv')
    data.columns = ['user_id', 'behavior', 'blogger_id', 'timestamp']
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    print("CSV数据加载成功。")
    print("附件1数据形状：", data.shape)
    print("附件1数据前5行：")
    print(data.head())
    print("数据日期范围：", data['timestamp'].min(), "至", data['timestamp'].max())
    print("唯一博主数：", data['blogger_id'].nunique())
    print("唯一用户数：", data['user_id'].nunique())
    print("行为类型：", data['behavior'].unique())
    with open(os.path.join(output_dir, '数据概览.txt'), 'w', encoding='utf-8') as f:
        f.write(f"附件1数据形状：{data.shape}\n")
        f.write(f"数据日期范围：{data['timestamp'].min()} 至 {data['timestamp'].max()}\n")
        f.write(f"唯一博主数：{data['blogger_id'].nunique()}\n")
        f.write(f"唯一用户数：{data['user_id'].nunique()}\n")
        f.write(f"行为类型：{data['behavior'].unique()}\n")
    print("数据概览已保存为 '输出结果/数据概览.txt'")
except FileNotFoundError:
    print("错误：未找到attachment1.csv，请确保文件位于脚本目录：D:\\010\\master\\代做\\数学建模\\C\\2025-51MCM-Problem C\\")
    exit()
except Exception as e:
    print(f"加载CSV文件时出错：{e}")
    exit()

# 为点赞、评论、关注行为添加隐含的观看行为
additional_views = data[data['behavior'].isin([2, 3, 4])][['user_id', 'blogger_id', 'timestamp']].copy()
additional_views['behavior'] = 1
data = pd.concat([data, additional_views], ignore_index=True)
print("隐含观看行为添加完成。")
print("附件1更新后形状：", data.shape)

# 步骤2：特征工程
print("\n步骤2：进行特征工程...")


def aggregate_features(df):
    features = df.groupby('blogger_id').apply(lambda x: pd.Series({
        'views': (x['behavior'] == 1).sum(),
        'likes': (x['behavior'] == 2).sum(),
        'comments': (x['behavior'] == 3).sum(),
        'follows': (x['behavior'] == 4).sum(),
        'unique_users': x['user_id'].nunique(),
        'avg_interactions_per_user': x['user_id'].count() / x['user_id'].nunique() if x['user_id'].nunique() > 0 else 0,
        'like_rate': (x['behavior'] == 2).sum() / (x['behavior'] == 1).sum() if (x['behavior'] == 1).sum() > 0 else 0,
        'comment_rate': (x['behavior'] == 3).sum() / (x['behavior'] == 1).sum() if (x['behavior'] == 1).sum() > 0 else 0
    })).reset_index()
    return features


data['date'] = data['timestamp'].dt.date
daily_data = {date: group for date, group in data.groupby('date')}
print("数据覆盖的日期：", sorted(daily_data.keys()))
with open(os.path.join(output_dir, '日期覆盖.txt'), 'w', encoding='utf-8') as f:
    f.write(f"数据覆盖的日期：{sorted(daily_data.keys())}\n")
print("日期覆盖已保存为 '输出结果/日期覆盖.txt'")

daily_features = []
for date, df in daily_data.items():
    features = aggregate_features(df)
    features['date'] = date
    daily_features.append(features)

if not daily_features:
    print("错误：无法生成每日特征，可能是日期分组为空。")
    print("建议：检查attachment1.csv的timestamp列是否包含有效日期。")
    exit()

features_df = pd.concat(daily_features, ignore_index=True)
print("特征工程完成。特征数据形状：", features_df.shape)
print("特征数据前5行：")
print(features_df.head())
print("特征数据博主数：", features_df['blogger_id'].nunique())
with open(os.path.join(output_dir, '特征数据概览.txt'), 'w', encoding='utf-8') as f:
    f.write(f"特征数据形状：{features_df.shape}\n")
    f.write(f"特征数据博主数：{features_df['blogger_id'].nunique()}\n")
    f.write("特征数据前5行：\n")
    f.write(f"{features_df.head().to_string()}\n")
print("特征数据概览已保存为 '输出结果/特征数据概览.txt'")

# 步骤3：相关性分析
print("\n步骤3：进行相关性分析...")
correlation_matrix = features_df[['views', 'likes', 'comments', 'follows', 'unique_users',
                                  'avg_interactions_per_user', 'like_rate', 'comment_rate']].corr()
print("相关性矩阵：")
print(correlation_matrix)
with open(os.path.join(output_dir, '相关性矩阵.txt'), 'w', encoding='utf-8') as f:
    f.write("相关性矩阵：\n")
    f.write(f"{correlation_matrix.to_string()}\n")
print("相关性矩阵已保存为 '输出结果/相关性矩阵.txt'")

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': '相关系数'})
plt.title('特征相关性矩阵')
plt.savefig(os.path.join(output_dir, '相关性矩阵_二维.png'))
plt.show()
print("2D相关性矩阵已保存为 '输出结果/相关性矩阵_二维.png'")

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(len(correlation_matrix)), range(len(correlation_matrix)))
Z = correlation_matrix.values
with sns.axes_style("whitegrid"):
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm')
ax.set_xticks(range(len(correlation_matrix)))
ax.set_yticks(range(len(correlation_matrix)))
ax.set_xticklabels(correlation_matrix.columns, rotation=45)
ax.set_yticklabels(correlation_matrix.index)
ax.set_zlabel('相关系数')
plt.title('3D特征相关性热力图')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.savefig(os.path.join(output_dir, '相关性矩阵_三维.png'))
plt.show()
print("3D相关性热力图已保存为 '输出结果/相关性矩阵_三维.png'")

# 步骤4：准备建模数据
print("\n步骤4：准备建模数据...")
features_df['date'] = pd.to_datetime(features_df['date'])
features_df = features_df.sort_values(['blogger_id', 'date'])
all_dates = pd.date_range(start=features_df['date'].min(), end=features_df['date'].max(), freq='D')
all_bloggers = features_df['blogger_id'].unique()
full_index = pd.MultiIndex.from_product([all_bloggers, all_dates], names=['blogger_id', 'date'])
full_df = pd.DataFrame(index=full_index).reset_index()
full_df['date'] = pd.to_datetime(full_df['date'])
features_df = full_df.merge(features_df, on=['blogger_id', 'date'], how='left').fillna(0)
features_df['next_day_follows'] = features_df.groupby('blogger_id')['follows'].shift(-1)
model_data = features_df.dropna(subset=['next_day_follows'])
print("模型数据形状：", model_data.shape)
print("模型数据日期范围：", model_data['date'].min(), "至", model_data['date'].max())
print("模型数据博主数：", model_data['blogger_id'].nunique())
blogger_days = model_data.groupby('blogger_id')['date'].count()
print("每位博主的数据天数：")
print(blogger_days)
with open(os.path.join(output_dir, '博主数据天数.txt'), 'w', encoding='utf-8') as f:
    f.write("每位博主的数据天数：\n")
    f.write(f"{blogger_days.to_string()}\n")
print("博主数据天数已保存为 '输出结果/博主数据天数.txt'")

feature_columns = ['views', 'likes', 'comments', 'unique_users', 'avg_interactions_per_user', 'like_rate', 'comment_rate']


def create_sequences(data, min_seq_length=1):
    sequences = []
    targets = []
    bloggers = data['blogger_id'].unique()
    print(f"总博主数：{len(bloggers)}")
    for blogger in bloggers:
        blogger_data = data[data['blogger_id'] == blogger].sort_values('date')
        num_days = len(blogger_data)
        if num_days >= min_seq_length:
            for i in range(num_days - min_seq_length + 1):
                seq = blogger_data[feature_columns].iloc[i:i + min_seq_length].values
                target = blogger_data['next_day_follows'].iloc[i + min_seq_length - 1]
                sequences.append(seq)
                targets.append(target)
        else:
            print(f"博主 {blogger} 数据不足，仅有 {num_days} 天，跳过。")
    sequences = np.array(sequences)
    targets = np.array(targets)
    print(f"生成序列数：{len(sequences)}")
    return sequences, targets


X, y = create_sequences(model_data, min_seq_length=1)
if len(X) == 0:
    print("错误：无法生成任何序列，可能是数据不足或博主数据天数少于1天。")
    print("建议：1. 检查attachment1.csv是否包含有效数据；2. 提供数据样本以进一步调试。")
    exit()

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("数据集划分：训练集：", X_train.shape, "验证集：", X_val.shape, "测试集：", X_test.shape)
with open(os.path.join(output_dir, '数据集划分.txt'), 'w', encoding='utf-8') as f:
    f.write(f"训练集形状：{X_train.shape}\n")
    f.write(f"验证集形状：{X_val.shape}\n")
    f.write(f"测试集形状：{X_test.shape}\n")
print("数据集划分信息已保存为 '输出结果/数据集划分.txt'")

# 步骤5：训练MLP+LSTM+Transformer模型
print("\n步骤5：训练MLP+LSTM+Transformer模型...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPLSTMTransformerModel(input_dim=len(feature_columns)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
early_stopping = EarlyStopping(patience=20)
num_epochs = 500
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print(f"轮次 {epoch + 1}/{num_epochs}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")
    scheduler.step(val_loss)
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("早停触发，恢复最佳模型...")
        model.load_state_dict(early_stopping.best_model_state)
        break

with open(os.path.join(output_dir, '训练过程.txt'), 'w', encoding='utf-8') as f:
    f.write("轮次,训练损失,验证损失\n")
    for i, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
        f.write(f"{i + 1},{t_loss:.4f},{v_loss:.4f}\n")
print("训练过程已保存为 '输出结果/训练过程.txt'")
loss_df = pd.DataFrame({'轮次': range(1, len(train_losses) + 1), '训练损失': train_losses, '验证损失': val_losses})
plt.figure(figsize=(12, 5))
sns.lineplot(data=loss_df.melt('轮次', var_name='损失类型', value_name='损失'), x='轮次', y='损失', hue='损失类型')
plt.title('模型损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.savefig(os.path.join(output_dir, '训练过程.png'))
plt.show()
print("训练过程图已保存为 '输出结果/训练过程.png'")

# 测试集评估
model.eval()
test_loss = 0
y_pred = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()
        y_pred.extend(outputs.cpu().numpy().flatten())
test_loss /= len(test_loader)
print(f"测试集损失（MSE）：{test_loss:.4f}")
y_pred = np.array(y_pred)
y_test_np = y_test
mre = np.mean((y_test_np - y_pred) / (y_test_np + 1e-10)) * 100
mare = np.mean(np.abs((y_test_np - y_pred) / (y_test_np + 1e-10))) * 100
print(f"平均相对误差（MRE）：{mre:.2f}%")
print(f"平均绝对相对误差（MARE）：{mare:.2f}%")
with open(os.path.join(output_dir, '测试评估.txt'), 'w', encoding='utf-8') as f:
    f.write(f"测试集损失（MSE）：{test_loss:.4f}\n")
    f.write(f"平均相对误差（MRE）：{mre:.2f}%\n")
    f.write(f"平均绝对相对误差（MARE）：{mare:.2f}%\n")
print("测试集评估已保存为 '输出结果/测试评估.txt'")

# 步骤6：预测7月21日新增关注数
print("\n步骤6：预测2024年7月21日新增关注数...")


def create_prediction_sequences(data, seq_length=1):
    sequences = []
    blogger_ids = []
    bloggers = data['blogger_id'].unique()
    print(f"预测阶段总博主数：{len(bloggers)}")
    for blogger in bloggers:
        blogger_data = data[data['blogger_id'] == blogger].sort_values('date').tail(seq_length)
        if len(blogger_data) == seq_length:
            seq = blogger_data[feature_columns].values
            sequences.append(seq)
            blogger_ids.append(blogger)
        else:
            print(f"博主 {blogger} 数据不足，仅有 {len(blogger_data)} 天，跳过预测。")
    sequences = np.array(sequences)
    blogger_ids = np.array(blogger_ids)
    print(f"生成预测序列数：{len(sequences)}")
    return sequences, blogger_ids


X_july20, blogger_ids = create_prediction_sequences(features_df, seq_length=1)
if len(X_july20) == 0:
    print("错误：无法生成预测序列，可能是博主数据不足。")
    print("建议：检查attachment1.csv是否包含有效数据。")
    exit()

X_july20_scaled = scaler.transform(X_july20.reshape(-1, X_july20.shape[-1])).reshape(X_july20.shape)
X_july20_tensor = torch.tensor(X_july20_scaled, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    predictions = model(X_july20_tensor).cpu().numpy().flatten()

# 生成所有博主的预测表
all_predictions_df = pd.DataFrame({
    'blogger_id': blogger_ids,
    'predicted_new_follows': np.round(predictions).astype(int)
})
print("表1：2024年7月21日所有博主新增关注数预测")
print(all_predictions_df)
all_predictions_df.to_csv(os.path.join(output_dir, '表1_所有博主预测.csv'), index=False, encoding='utf-8-sig')
with open(os.path.join(output_dir, '表1_所有博主预测.txt'), 'w', encoding='utf-8') as f:
    f.write("表1：2024年7月21日所有博主新增关注数预测：\n")
    f.write(f"{all_predictions_df.to_string()}\n")
print("所有博主预测表已保存为 '输出结果/表1_所有博主预测.csv' 和 '输出结果/表1_所有博主预测.txt'")

# 生成前5位博主的预测表
top_5_df = all_predictions_df.nlargest(5, 'predicted_new_follows')[['blogger_id', 'predicted_new_follows']]
print("\n表2：2024年7月21日新增关注数最多的前5位博主")
print(top_5_df)
top_5_df.to_csv(os.path.join(output_dir, '表2_前五博主.csv'), index=False, encoding='utf-8-sig')
with open(os.path.join(output_dir, '表2_前五博主.txt'), 'w', encoding='utf-8') as f:
    f.write("表2：2024年7月21日新增关注数最多的前5位博主：\n")
    f.write(f"{top_5_df.to_string()}\n")
print("前5位博主预测表已保存为 '输出结果/表2_前五博主.csv' 和 '输出结果/表2_前五博主.txt'")

# 步骤7：回归分析可视化
print("\n步骤7：可视化回归分析...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='理想预测线')
plt.xlabel('实际新增关注数')
plt.ylabel('预测新增关注数')
plt.title('回归分析：测试集实际 vs 预测')
plt.legend()
plt.savefig(os.path.join(output_dir, '回归分析.png'))
plt.show()
print("回归分析图已保存为 '输出结果/回归分析.png'")
with open(os.path.join(output_dir, '回归分析数据.txt'), 'w', encoding='utf-8') as f:
    f.write("回归分析数据（测试集实际 vs 预测）：\n")
    f.write("实际值,预测值\n")
    for actual, pred in zip(y_test, y_pred):
        f.write(f"{actual:.4f},{pred:.4f}\n")
print("回归分析数据已保存为 '输出结果/回归分析数据.txt'")