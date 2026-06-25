import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 1. 数据加载
print("正在加载数据...")
try:
    data = pd.read_csv('Attachment 1.csv')
except FileNotFoundError:
    print("错误：未找到 Attachment 1.csv，请确保文件存在！")
    exit()

data.columns = ['User ID', 'User behaviour', 'Blogger ID', 'Time']
data['Time'] = pd.to_datetime(data['Time'])
data = data[(data['Time'] >= '2024-07-11') & (data['Time'] <= '2024-07-20')]
print(f"数据加载完成！共 {len(data)} 条记录")

# 筛选目标用户
target_users = ['U10', 'U1951', 'U1833', 'U26447']
data = data[data['User ID'].isin(target_users)]
print(f"筛选出目标用户数据：{len(data)} 条记录")

# 2. 特征工程
print("正在进行特征工程...")
# 时间特征
data['Hour'] = data['Time'].dt.hour
data['Day'] = data['Time'].dt.day
data['Time_Diff'] = data.groupby('User ID')['Time'].diff().dt.total_seconds().fillna(0) / 3600

# 活跃时间窗口（最近3天行为密度）
data['Time_Since_Start'] = (data['Time'] - pd.to_datetime('2024-07-11')).dt.total_seconds() / 86400
user_activity = data.groupby('User ID').apply(
    lambda x: x[x['Time'] >= x['Time'].max() - pd.Timedelta(days=3)]['User behaviour'].count() / 3
).reset_index(name='Recent_Activity')
data = data.merge(user_activity, on='User ID')

# 用户和博主统计
user_stats = data.groupby('User ID').agg({
    'User behaviour': ['count', lambda x: (x == 2).sum(), lambda x: (x == 3).sum(), lambda x: (x == 4).sum()]
}).reset_index()
user_stats.columns = ['User ID', 'Behavior_Count', 'Like_Count', 'Comment_Count', 'Follow_Count']
data = data.merge(user_stats, on='User ID')

blogger_stats = data.groupby('Blogger ID').agg({
    'User behaviour': ['count', lambda x: (x == 2).sum(), lambda x: (x == 3).sum(), lambda x: (x == 4).sum()]
}).reset_index()
blogger_stats.columns = ['Blogger ID', 'Blogger_Behavior_Count', 'Blogger_Like_Count', 'Blogger_Comment_Count', 'Blogger_Follow_Count']
data = data.merge(blogger_stats, on='Blogger ID')

# 用户-博主互动频率
user_blogger_interactions = data.groupby(['User ID', 'Blogger ID'])['User behaviour'].count().reset_index()
user_blogger_interactions.columns = ['User ID', 'Blogger ID', 'User_Blogger_Interaction_Count']
data = data.merge(user_blogger_interactions, on=['User ID', 'Blogger ID'], how='left')
data['User_Blogger_Interaction_Count'] = data['User_Blogger_Interaction_Count'].fillna(0)

# 编码
user_encoder = {u: i for i, u in enumerate(target_users)}
blogger_ids = data['Blogger ID'].unique()
blogger_encoder = {b: i for i, b in enumerate(blogger_ids)}
data['User ID Encoded'] = data['User ID'].map(user_encoder)
data['Blogger ID Encoded'] = data['Blogger ID'].map(blogger_encoder)

# 行为独热编码
behavior_dummies = pd.get_dummies(data['User behaviour'], prefix='Behavior')
data = pd.concat([data, behavior_dummies], axis=1)
for i in range(1, 5):
    col = f'Behavior_{i}'
    if col not in data.columns:
        data[col] = 0

# 构造时间序列
sequence_length = 3
features = ['User ID Encoded', 'Blogger ID Encoded', 'Hour', 'Day', 'Time_Diff', 'Recent_Activity',
            'Behavior_Count', 'Like_Count', 'Comment_Count', 'Follow_Count',
            'Blogger_Behavior_Count', 'Blogger_Like_Count', 'Blogger_Comment_Count', 'Blogger_Follow_Count',
            'User_Blogger_Interaction_Count', 'Behavior_1', 'Behavior_2', 'Behavior_3', 'Behavior_4']
X_sequences = []
y_online = []
y_interactions = []

for user in tqdm(target_users, desc="构造时间序列"):
    user_data = data[data['User ID'] == user].sort_values('Time')
    print(f"用户 {user} 数据量：{len(user_data)} 条")
    if len(user_data) < sequence_length:
        print(f"用户 {user} 数据不足 {sequence_length} 条，跳过")
        continue
    for i in range(len(user_data) - sequence_length):
        seq = user_data[features].iloc[i:i+sequence_length].values
        current_time = user_data.iloc[i+sequence_length-1]['Time']
        future_data = user_data[user_data['Time'] > current_time]
        # 在线定义：未来24小时有任意Behavior_1,2,3,4
        online_prob = 1.0 if any(future_data['Time'] <= current_time + pd.Timedelta(hours=24)) else 0.0
        interactions = user_data.iloc[i+sequence_length-1][['Behavior_2', 'Behavior_3', 'Behavior_4']].sum()
        X_sequences.append(seq)
        y_online.append(online_prob)
        y_interactions.append(interactions)

if len(X_sequences) == 0:
    print("错误：所有用户数据不足，无法构造序列！")
    exit()

X_sequences = np.array(X_sequences)
y_online = np.array(y_online)
y_interactions = np.array(y_interactions)
y_interactions = np.clip(y_interactions, 0, 5)
print(f"构造序列完成：{len(X_sequences)} 个序列")
print(f"在线概率分布：{np.histogram(y_online, bins=10)[1]}")
print(f"互动数分布：{np.histogram(y_interactions, bins=10)[1]}")

# 绘制时序逻辑判断图
print("正在生成时序逻辑判断图...")
for user in target_users:
    user_data = data[data['User ID'] == user].sort_values('Time')
    if len(user_data) < sequence_length:
        continue
    plt.figure(figsize=(12, 4))
    for behavior in [1, 2, 3, 4]:
        behavior_data = user_data[user_data['User behaviour'] == behavior]
        plt.scatter(behavior_data['Time'], [behavior] * len(behavior_data), label=f'Behavior {behavior}', alpha=0.6)
    plt.axvline(pd.to_datetime('2024-07-20'), color='r', linestyle='--', label='Prediction Point')
    plt.xlabel('Time')
    plt.ylabel('Behavior Type')
    plt.title(f'Timeline of Behaviors for User {user}')
    plt.legend()
    plt.savefig(f'user_timeline_{user}.png')
    plt.close()

# 数据标准化
scaler = StandardScaler()
X_sequences = scaler.fit_transform(X_sequences.reshape(-1, X_sequences.shape[-1])).reshape(X_sequences.shape)

# 划分数据集
X_train, X_test, y_online_train, y_online_test, y_interactions_train, y_interactions_test = train_test_split(
    X_sequences, y_online, y_interactions, test_size=0.2, random_state=42
)
print("数据预处理完成！")

# 3. 回归分析
print("正在进行回归分析...")
regression_data = data[['Behavior_1', 'Behavior_2', 'Behavior_3', 'Behavior_4']].copy()
correlation_matrix = regression_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix with Behavior_4 (Follow) as Dependent Variable")
plt.savefig('correlation_matrix.png')
plt.close()

X_reg = regression_data[['Behavior_1', 'Behavior_2', 'Behavior_3']]
y_reg = regression_data['Behavior_4']
log_reg = LogisticRegression()
log_reg.fit(X_reg, y_reg)
print("回归分析完成！")
print("逻辑回归系数：", log_reg.coef_)

# 4. 定义GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc_online = nn.Linear(hidden_size, 1)
        self.fc_interactions = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]
        gru_out = self.bn(gru_out)
        online_pred = self.sigmoid(self.fc_online(gru_out))
        interactions_pred = self.fc_interactions(gru_out)
        return online_pred, interactions_pred

# 模型参数
input_size = X_sequences.shape[-1]
hidden_size = 16
num_layers = 1
model = GRUModel(input_size, hidden_size, num_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 损失函数和优化器
criterion_online = nn.MSELoss()
criterion_interactions = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# 5. 模型训练
print("开始模型训练...")
num_epochs = 250
train_losses = []
test_losses = []
best_loss = float('inf')
patience = 20
early_stop_counter = 0

for epoch in tqdm(range(num_epochs), desc="训练进度"):
    model.train()
    epoch_loss = 0
    for i in range(0, len(X_train), 16):
        X_batch = torch.tensor(X_train[i:i+16], dtype=torch.float32).to(device)
        y_online_batch = torch.tensor(y_online_train[i:i+16], dtype=torch.float32).to(device)
        y_interactions_batch = torch.tensor(y_interactions_train[i:i+16], dtype=torch.float32).to(device)

        optimizer.zero_grad()
        online_pred, interactions_pred = model(X_batch)
        loss_online = criterion_online(online_pred.squeeze(), y_online_batch)
        weights = torch.where(y_interactions_batch > 0, 2.0, 1.0).to(device)
        loss_interactions = (criterion_interactions(interactions_pred.squeeze(), y_interactions_batch) * weights).mean()
        loss = 0.3 * loss_online + 0.7 * loss_interactions
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / (len(X_train) // 16 + 1))

    # 测试集评估
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_online_test_tensor = torch.tensor(y_online_test, dtype=torch.float32).to(device)
        y_interactions_test_tensor = torch.tensor(y_interactions_test, dtype=torch.float32).to(device)
        online_pred, interactions_pred = model(X_test_tensor)
        loss_online = criterion_online(online_pred.squeeze(), y_online_test_tensor)
        loss_interactions = criterion_interactions(interactions_pred.squeeze(), y_interactions_test_tensor)
        test_loss = 0.3 * loss_online + 0.7 * loss_interactions
        test_losses.append(test_loss.item())

    scheduler.step(test_loss)
    if test_loss < best_loss:
        best_loss = test_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("早停触发，停止训练")
            break

print("模型训练完成！")

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss Curve')
plt.legend()
plt.savefig('loss_curve.png')
plt.close()

# 绘制预测 vs 真实值散点图
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    online_pred, interactions_pred = model(X_test_tensor)
    online_pred = online_pred.cpu().numpy().squeeze()
    interactions_pred = interactions_pred.cpu().numpy().squeeze()

plt.figure(figsize=(8, 6))
plt.scatter(y_interactions_test, interactions_pred, alpha=0.5)
plt.plot([y_interactions_test.min(), y_interactions_test.max()], [y_interactions_test.min(), y_interactions_test.max()], 'r--')
plt.xlabel('True Interactions')
plt.ylabel('Predicted Interactions')
plt.title('Predicted vs True Interactions')
plt.savefig('interactions_scatter.png')
plt.close()

# 绘制在线概率和互动数分布
plt.figure(figsize=(8, 6))
plt.hist(online_pred, bins=20, alpha=0.5, label='Predicted Online Probability')
plt.hist(y_online_test, bins=20, alpha=0.5, label='True Online Probability')
plt.xlabel('Online Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted vs True Online Probability')
plt.legend()
plt.savefig('online_probability_distribution.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(interactions_pred, bins=20, alpha=0.5, label='Predicted Interactions')
plt.hist(y_interactions_test, bins=20, alpha=0.5, label='True Interactions')
plt.xlabel('Interactions')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted vs True Interactions')
plt.legend()
plt.savefig('interactions_distribution.png')
plt.close()

# 6. 模型评估
print("正在评估模型...")
auc = roc_auc_score(y_online_test, online_pred) if len(np.unique(y_online_test)) > 1 else 0.0
mre = np.mean(np.abs(interactions_pred - y_interactions_test) / (y_interactions_test + 1e-10))
mare = np.mean(np.abs(interactions_pred - y_interactions_test) / (np.abs(y_interactions_test) + 1e-10))
r2 = r2_score(y_interactions_test, interactions_pred)
mae = mean_absolute_error(y_interactions_test, interactions_pred)
mse = mean_squared_error(y_interactions_test, interactions_pred)

eval_df = pd.DataFrame({
    'Metric': ['AUC (Online)', 'MRE (Interactions)', 'MARE (Interactions)', 'R² (Interactions)', 'MAE (Interactions)', 'MSE (Interactions)'],
    'Value': [auc * 100, mre * 100, mare * 100, r2 * 100, mae * 100, mse * 100]
})
print("\n模型评估结果（百分比）：")
print(eval_df)
eval_df.to_csv('evaluation_metrics.csv', index=False)

# 7. 预测2024.7.21
print("正在预测2024.7.21的用户行为...")
results = []
for user in tqdm(target_users, desc="预测进度"):
    user_data = data[data['User ID'] == user].sort_values('Time')
    if len(user_data) < sequence_length:
        print(f"用户 {user} 数据不足，预测为不在线")
        results.append([user, 0, 0, '', '', ''])
        continue

    seq = user_data[features].iloc[-sequence_length:].values
    seq = scaler.transform(seq.reshape(-1, seq.shape[-1])).reshape(1, sequence_length, -1)
    seq_tensor = torch.tensor(seq, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        online_pred, interactions = model(seq_tensor)
        online_prob = online_pred.cpu().numpy().squeeze()
        interactions = interactions.cpu().numpy().squeeze()

    if online_prob > 0.5:
        blogger_interactions = []
        for blogger in blogger_ids[:100]:
            blogger_seq = seq.copy()
            blogger_seq[:, :, features.index('Blogger ID Encoded')] = blogger_encoder[blogger]
            blogger_seq_tensor = torch.tensor(blogger_seq, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, blogger_interactions_pred = model(blogger_seq_tensor)
            blogger_interactions.append((blogger, blogger_interactions_pred.cpu().numpy().squeeze()))

        blogger_interactions.sort(key=lambda x: x[1], reverse=True)
        top_bloggers = blogger_interactions[:3]
        results.append([
            user, 1, interactions,
            top_bloggers[0][0] if len(top_bloggers) > 0 else '',
            top_bloggers[1][0] if len(top_bloggers) > 1 else '',
            top_bloggers[2][0] if len(top_bloggers) > 2 else ''
        ])
    else:
        results.append([user, 0, 0, '', '', ''])

result_df = pd.DataFrame(results, columns=[
    'User ID', 'Online (1=Yes, 0=No)', 'Predicted Interactions',
    'Top Blogger 1', 'Top Blogger 2', 'Top Blogger 3'
])
print("\n预测结果：")
print(result_df)
result_df.to_csv('prediction_results.csv', index=False)
print("预测结果已保存至 prediction_results.csv")