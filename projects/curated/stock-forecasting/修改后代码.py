import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, confusion_matrix
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 1. 数据集类 ---
class StockDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_level=0.01):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.augment = augment
        self.noise_level = noise_level

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]

        if self.augment:
            # 微弱噪音增强
            noise = torch.randn_like(x_sample) * self.noise_level
            x_sample = x_sample + noise
        return x_sample, y_sample


# --- 2. 模型结构 (CNN + Attention) ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class EnhancedCNN(nn.Module):
    def __init__(self, input_features, dropout_rate=0.4):
        super(EnhancedCNN, self).__init__()

        # 输入层
        self.stem = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1)
        )

        # 核心卷积层
        self.layer1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        self.se1 = SEBlock(64)
        self.pool1 = nn.MaxPool1d(2)

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        self.se2 = SEBlock(128)
        self.pool2 = nn.MaxPool1d(2)

        self.gap = nn.AdaptiveAvgPool1d(1)

        # 分类头
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (Batch, Window, Features) -> (Batch, Features, Window)
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.se1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.pool2(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.head(x)
        return x


# --- 3. 特征工程 (平稳化处理) ---
def engineer_features(df):
    data = df.copy()

    # 基础保护
    if 'volume' not in data.columns: data['volume'] = 1.0

    # 1. 对数收益率
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))

    # 2. 相对波动率
    data['volatility'] = data['log_ret'].rolling(10).std()

    # 3. RSI (0-1)
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    data['rsi'] = calc_rsi(data['close']) / 100.0

    # 4. MACD Ratio (MACD / Close)
    ema12 = data['close'].ewm(span=12).mean()
    ema26 = data['close'].ewm(span=26).mean()
    data['macd_norm'] = (ema12 - ema26) / data['close']

    # 5. ATR Ratio
    high_low = data['high'] - data['low']
    tr = high_low.rolling(14).mean()
    data['atr_norm'] = tr / data['close']

    # 6. Volume Trend
    vol_ma = data['volume'].rolling(20).mean()
    data['vol_rel'] = np.log((data['volume'] + 1) / (vol_ma + 1))

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    return data


# --- 4. 数据准备 (含动态标签) ---
def prepare_data(df, window_size=60):
    # 关键修改：动态分位数标签 (Quantile Labeling)
    # 解决 600028 等大盘股波动小、无法触发固定阈值的问题

    # 预测未来 5 天收益
    df['future_ret'] = df['close'].shift(-5) / df['close'] - 1
    df.dropna(subset=['future_ret'], inplace=True)

    # --- 【修改】动态计算阈值：涨幅前 35% 为 1，跌幅前 35% 为 0，中间 30% 震荡丢弃 ---
    up_threshold = df['future_ret'].quantile(0.65)
    down_threshold = df['future_ret'].quantile(0.35)

    # 防止阈值过于接近 (极度横盘时)
    if up_threshold < 0.005: up_threshold = 0.005
    if down_threshold > -0.005: down_threshold = -0.005

    print(f"  Dynamic Thresholds -> Up: >{up_threshold:.4f}, Down: <{down_threshold:.4f}")

    df['label'] = np.nan
    df.loc[df['future_ret'] > up_threshold, 'label'] = 1
    df.loc[df['future_ret'] < down_threshold, 'label'] = 0

    # 只保留有标签的数据
    df_clean = df.dropna(subset=['label']).copy()

    feature_cols = ['log_ret', 'volatility', 'rsi', 'macd_norm', 'atr_norm', 'vol_rel']
    data_vals = df_clean[feature_cols].values

    # Robust Scaling (Clip极值)
    data_vals = np.clip(data_vals, -5, 5)

    y_vals = df_clean['label'].values
    dates = df_clean['date'].values

    X, y, d = [], [], []
    for i in range(len(data_vals) - window_size):
        X.append(data_vals[i: i + window_size])
        y.append(y_vals[i + window_size])
        d.append(dates[i + window_size])

    return np.array(X), np.array(y), np.array(d)


# --- 5. 训练主流程 ---
def train_stock(file_path, output_dir):
    stock_code = os.path.basename(file_path).split('_')[0]

    # 读取清洗后的数据
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 只有当数据量足够时才处理
    if len(df) < 500: return

    # 特征工程
    df = engineer_features(df)

    # 准备数据 (Window=60)
    X, y, dates = prepare_data(df, window_size=60)

    if len(X) < 100:
        print(f"  {stock_code}: Not enough samples after filtering.")
        return

    # 划分数据集 (最后 15% 做测试)
    split_idx = int(len(X) * 0.85)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # --- 【新增】保存测试集日期，用于后续输出预测结果 ---
    dates_test = dates[split_idx:]

    print(f"Stock: {stock_code} | Total: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")

    # 关键修正：添加 drop_last=True，防止最后一个 batch 大小为 1 导致 BatchNorm 报错
    train_loader = DataLoader(StockDataset(X_train, y_train, augment=True), batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(StockDataset(X_test, y_test, augment=False), batch_size=32, shuffle=False)

    # 初始化模型
    model = EnhancedCNN(input_features=X.shape[2]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.BCELoss()

    best_loss = float('inf')
    best_auc = 0.5
    patience = 0

    # 训练循环
    for epoch in range(40):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                out = model(bx)
                preds.extend(out.cpu().numpy().flatten())
                targets.extend(by.numpy().flatten())

        preds = np.array(preds)
        targets = np.array(targets)

        try:
            val_auc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.5
        except:
            val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            # 保存最佳权重 (可选)
        else:
            patience += 1
            if patience >= 10: break

    # --- 最终计算指标 (AUC, ACC, MCC) ---
    pred_labels = (preds > 0.5).astype(int)

    # 1. AUC
    try:
        final_auc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.5
    except:
        final_auc = 0.5

    # 2. Accuracy
    final_acc = accuracy_score(targets, pred_labels)

    # 3. MCC (Matthews Correlation Coefficient)
    final_mcc = matthews_corrcoef(targets, pred_labels)

    print(f"  -> Result: AUC: {final_auc:.4f} | ACC: {final_acc:.4f} | MCC: {final_mcc:.4f}")

    # --- 【新增】保存个股预测结果到CSV ---
    predictions_dir = os.path.join(output_dir, 'predictions')
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    pred_df = pd.DataFrame({
        'Date': dates_test,
        'True_Label': targets,
        'Pred_Prob': preds,
        'Pred_Label': pred_labels
    })
    pred_df.to_csv(os.path.join(predictions_dir, f'{stock_code}_preds.csv'), index=False)

    # 保存综合指标
    results_path = os.path.join(output_dir, 'final_metrics.csv')
    res_df = pd.DataFrame([{
        'Stock': stock_code,
        'AUC': final_auc,
        'ACC': final_acc,
        'MCC': final_mcc,
        'Samples': len(targets)
    }])

    if not os.path.exists(results_path):
        res_df.to_csv(results_path, index=False)
    else:
        res_df.to_csv(results_path, index=False, mode='a', header=False)


if __name__ == "__main__":
    # 确保这里的文件夹路径和你 clean_data.py 输出的一致
    data_dir = 'data_v3'
    output_dir = 'Data10_Result'

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if os.path.exists(os.path.join(output_dir, 'final_metrics.csv')):
        os.remove(os.path.join(output_dir, 'final_metrics.csv'))

    files = glob.glob(os.path.join(data_dir, "*_clean.csv"))
    if not files:
        # 兼容旧文件名
        files = glob.glob(os.path.join(data_dir, "*.csv"))

    print(f"Found {len(files)} files to train.")

    # 1. 训练所有文件
    for f in files:
        try:
            train_stock(f, output_dir)
        except Exception as e:
            print(f"Error {f}: {e}")

    # 2. 打印最终汇总结果
    final_csv_path = os.path.join(output_dir, 'final_metrics.csv')
    if os.path.exists(final_csv_path):
        print("\n" + "=" * 35)
        print("       FINAL AVERAGE RESULTS       ")
        print("=" * 35)

        df_res = pd.read_csv(final_csv_path)
        if not df_res.empty:
            avg_auc = df_res['AUC'].mean()
            avg_acc = df_res['ACC'].mean()
            avg_mcc = df_res['MCC'].mean()

            print(f"Processed Stocks : {len(df_res)}")
            print(f"Average AUC      : {avg_auc:.4f}")
            print(f"Average ACC      : {avg_acc:.4f}")
            print(f"Average MCC      : {avg_mcc:.4f}")

            # 统计一下 MCC > 0 的比例
            profitable_ratio = (df_res['MCC'] > 0.05).mean() * 100
            print(f"Stocks with MCC > 0.05: {profitable_ratio:.1f}%")
        else:
            print("Results file is empty.")
        print("=" * 35 + "\n")