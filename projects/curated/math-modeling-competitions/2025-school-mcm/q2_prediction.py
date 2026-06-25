import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib

# ---------------- 1. 配置 ----------------
TRAIN_START = "2024-07-11"
TRAIN_END   = "2024-07-20"
PRED_DAY    = "2024-07-22"
TARGET_USERS= ["U7","U6749","U5769","U14990","U52010"]
BATCH_SIZE  = 256
EPOCHS      = 50
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
os.makedirs("第二题的所有数据", exist_ok=True)

# ---------------- 2. 读取数据 ----------------
print("🔧 1. 读取 CSV…")
df1 = pd.read_csv("Attachment 1.csv", parse_dates=["时间 (Time)"])
df2 = pd.read_csv("Attachment 2.csv", parse_dates=["时间 (Time)"])

# ---------------- 3. 筛出训练时段 全量数据 ----------------
mask_train = (df1["时间 (Time)"] >= TRAIN_START) & (df1["时间 (Time)"] <= TRAIN_END)
hist_df   = df1[mask_train].copy()

# ---------------- 4. 特征工程 ----------------
pairs = hist_df.groupby(["用户ID (User ID)","博主ID (Blogger ID)"])
rows = []
for (uid, bid), grp in tqdm(pairs, desc="构造特征"):
    watch_cnt   = int((grp["用户行为 (User behaviour)"]==1).sum())
    like_cnt    = int((grp["用户行为 (User behaviour)"]==2).sum())
    comment_cnt = int((grp["用户行为 (User behaviour)"]==3).sum())
    follow      = int((grp["用户行为 (User behaviour)"]==4).any())
    last_time   = grp["时间 (Time)"].max()
    recency     = (pd.to_datetime(TRAIN_END) - last_time).days
    active_days = grp["时间 (Time)"].dt.date.nunique()
    rows.append([uid, bid, watch_cnt, like_cnt, comment_cnt, recency, active_days, follow])
feat_df = pd.DataFrame(rows, columns=["user","blogger","watch_cnt","like_cnt","comment_cnt","recency_days","active_days","follow"])

# ---------------- 5. 线性回归 ----------------
X_lr = feat_df[["watch_cnt","like_cnt","comment_cnt","recency_days","active_days"]].values
y_lr = feat_df["follow"].values
lr = LinearRegression().fit(X_lr, y_lr)

# ---------------- 6. PyTorch Dataset ----------------
class FollowDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(df[["watch_cnt","like_cnt","comment_cnt","recency_days","active_days"]].values, dtype=torch.float32)
        self.y = torch.tensor(df["follow"].values, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

loader = DataLoader(FollowDataset(feat_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ---------------- 7. 定义模型 ----------------
class MLP(nn.Module):
    def __init__(self, in_dim=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

model = MLP().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- 8. 训练 ----------------
train_losses = []
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    for xb, yb in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(loader.dataset)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")

# 保存训练损失图
plt.figure()
plt.plot(range(1, EPOCHS+1), train_losses)
plt.xlabel("训练轮数")
plt.ylabel("训练损失")
plt.title("训练损失曲线")
plt.savefig("第二题的所有数据/train_loss.png")
plt.close()

# ---------------- 9. 模型评估 ----------------
model.eval()
with torch.no_grad():
    X_all = torch.tensor(X_lr, dtype=torch.float32).to(DEVICE)
    y_true = y_lr
    y_pred = model(X_all).cpu().numpy().flatten()
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mre = np.mean(np.abs(y_pred - y_true) / (y_true + 1e-6))
mare = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-6))
print(f"MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, MRE={mre:.4f}, MARE={mare:.4f}")

# ---------------- 10. 预测数据处理 ----------------
pred_hist = hist_df.copy()
pred_today = df2[df2["时间 (Time)"].dt.date == pd.to_datetime(PRED_DAY).date()]
all_actions = pd.concat([pred_hist, pred_today], ignore_index=True)

pairs_p = all_actions.groupby(["用户ID (User ID)","博主ID (Blogger ID)"])
pred_rows = []
for (uid,bid), grp in tqdm(pairs_p, desc="构造预测特征"):
    if uid not in TARGET_USERS:
        continue
    w = int((grp["用户行为 (User behaviour)"]==1).sum())
    l = int((grp["用户行为 (User behaviour)"]==2).sum())
    c = int((grp["用户行为 (User behaviour)"]==3).sum())
    last = grp["时间 (Time)"].max()
    rec = (pd.to_datetime(PRED_DAY) - last).days
    days= grp["时间 (Time)"].dt.date.nunique()
    if ((hist_df["用户ID (User ID)"]==uid)&(hist_df["博主ID (Blogger ID)"]==bid)&(hist_df["用户行为 (User behaviour)"]==4)).any():
        continue
    pred_rows.append([uid,bid,w,l,c,rec,days])
pred_df = pd.DataFrame(pred_rows, columns=["user","blogger","watch_cnt","like_cnt","comment_cnt","recency_days","active_days"])

Xp = torch.tensor(pred_df[["watch_cnt","like_cnt","comment_cnt","recency_days","active_days"]].values, dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    probs = model(Xp).cpu().numpy().flatten()
pred_df["prob_follow"] = probs

results = pred_df[pred_df["prob_follow"]>=0.2][["user","blogger","prob_follow"]]
results.to_csv("predicted_new_follows_full_data.csv", index=False)

# ---------------- 11. 图表分析扩展 ----------------
plt.figure()
plt.hist(pred_df["prob_follow"], bins=50, color="skyblue", edgecolor="black")
plt.title("预测关注概率分布")
plt.xlabel("预测概率")
plt.ylabel("样本数")
plt.savefig("第二题的所有数据/pred_prob_hist.png")
plt.close()

# 散点图：每个特征 vs. follow
features = ["watch_cnt","like_cnt","comment_cnt","recency_days","active_days"]
for feat in features:
    plt.figure()
    plt.scatter(feat_df[feat], feat_df["follow"], alpha=0.3)
    plt.xlabel(feat)
    plt.ylabel("是否关注")
    plt.title(f"{feat} 与是否关注的关系")
    plt.savefig(f"第二题的所有数据/{feat}_vs_follow.png")
    plt.close()

# 线性回归系数条形图
plt.figure()
coefs = lr.coef_
plt.barh(features, coefs, color="coral")
plt.title("线性回归系数（特征重要性）")
plt.xlabel("权重")
plt.savefig("第二题的所有数据/linear_feature_importance.png")
plt.close()

# 用户分布图（Top-K）
top_k = results.sort_values("prob_follow", ascending=False).head(50)
if not top_k.empty:
    plt.figure(figsize=(10, 5))
    top_k["user"].value_counts().plot(kind="bar", color="green")
    plt.title("Top-K 新关注中各用户出现次数")
    plt.xlabel("用户ID")
    plt.ylabel("预测新增关注数")
    plt.tight_layout()
    plt.savefig("第二题的所有数据/topk_user_distribution.png")
    plt.close()

print("✅ 所有图表已保存至 第二题的所有数据/ 目录，预测结果保存至 predicted_new_follows_full_data.csv")
