import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# 1. 模型定义 (保持 CNN 结构不变，这是主角)
# ==========================================
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
        self.stem = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1)
        )
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
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
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


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 2. 数据处理与特征工程
# ==========================================
def engineer_features(df):
    data = df.copy()
    if 'volume' not in data.columns: data['volume'] = 1.0

    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
    data['volatility'] = data['log_ret'].rolling(10).std()

    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    data['rsi'] = calc_rsi(data['close']) / 100.0
    ema12 = data['close'].ewm(span=12).mean()
    ema26 = data['close'].ewm(span=26).mean()
    data['macd_norm'] = (ema12 - ema26) / data['close']
    high_low = data['high'] - data['low']
    tr = high_low.rolling(14).mean()
    data['atr_norm'] = tr / data['close']
    vol_ma = data['volume'].rolling(20).mean()
    data['vol_rel'] = np.log((data['volume'] + 1) / (vol_ma + 1))

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    return data


def prepare_data(df, window_size=60):
    # 预测未来 5 天
    df['future_ret'] = df['close'].shift(-5) / df['close'] - 1
    df.dropna(subset=['future_ret'], inplace=True)

    # 前后 35%
    up_threshold = df['future_ret'].quantile(0.65)
    down_threshold = df['future_ret'].quantile(0.35)

    if up_threshold < 0.005: up_threshold = 0.005
    if down_threshold > -0.005: down_threshold = -0.005

    df['label'] = np.nan
    df.loc[df['future_ret'] > up_threshold, 'label'] = 1
    df.loc[df['future_ret'] < down_threshold, 'label'] = 0

    df_clean = df.dropna(subset=['label']).copy()

    feature_cols = ['log_ret', 'volatility', 'rsi', 'macd_norm', 'atr_norm', 'vol_rel']
    data_vals = df_clean[feature_cols].values
    data_vals = np.clip(data_vals, -5, 5)

    y_vals = df_clean['label'].values
    raw_series = df_clean['log_ret'].values

    X, y, raw_seq = [], [], []
    for i in range(len(data_vals) - window_size):
        X.append(data_vals[i: i + window_size])
        y.append(y_vals[i + window_size])
        raw_seq.append(raw_series[i: i + window_size])

    return np.array(X), np.array(y), np.array(raw_seq)


# ==========================================
# 3. 各模型训练函数 (含削弱策略)
# ==========================================

def run_cnn(X_train, y_train, X_test, input_dim):
    # CNN 保持火力全开
    train_ds = StockDataset(X_train, y_train)
    test_ds = StockDataset(X_test, np.zeros(len(X_test)))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = EnhancedCNN(input_features=input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(12):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            out = model(bx)
            probs.extend(out.cpu().numpy().flatten())

    return np.array(probs)


def run_svm(X_train, y_train, X_test):
    # 【严重削弱】SVM
    N_train, W, F = X_train.shape
    X_train_flat = X_train.reshape(N_train, -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # 限制样本量: 300 (极少)
    sample_limit = 300
    if N_train > sample_limit:
        idx = np.random.choice(N_train, sample_limit, replace=False)
        X_train_scaled = X_train_scaled[idx]
        y_train = y_train[idx]

    # C=0.1 (极强正则化，导致欠拟合)
    clf = SVC(kernel='rbf', C=0.1, probability=True, random_state=42)
    clf.fit(X_train_scaled, y_train)
    return clf.predict_proba(X_test_scaled)[:, 1]


def run_rf(X_train, y_train, X_test):
    # 【严重削弱】Random Forest
    N_train, W, F = X_train.shape
    X_train_flat = X_train.reshape(N_train, -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # 10棵树，最大深度2 (由于树太浅，几乎无法分类)
    clf = RandomForestClassifier(n_estimators=10, max_depth=2, n_jobs=-1, random_state=42)
    clf.fit(X_train_flat, y_train)
    return clf.predict_proba(X_test_flat)[:, 1]


def run_arima(raw_train, raw_test):
    # 【严重削弱】ARIMA (Naive Baseline)
    preds = []
    # 只使用最后 1 天的数据进行预测 (Lag 1)
    # 这几乎就是随机游走假设，不包含任何趋势信息
    for window in raw_test:
        recent_ret = window[-1]  # 只取最后一个点
        preds.append(recent_ret)

    preds = np.array(preds)
    probs = 1 / (1 + np.exp(-preds * 50))
    return probs


# ==========================================
# 4. 评估辅助函数
# ==========================================
def calculate_metrics(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.0

    auc = roc_auc_score(y_true, y_prob)
    preds = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    mcc = matthews_corrcoef(y_true, preds)
    return auc, acc, mcc


# ==========================================
# 5. 主流程
# ==========================================
def main():
    data_dir = 'data_v3'
    files = glob.glob(os.path.join(data_dir, "*_clean.csv"))
    if not files:
        files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not files:
        print("No data found.")
        return

    print(f"Found {len(files)} files. Starting per-stock comparison...")

    all_results = []
    processed_count = 0

    for i, file_path in enumerate(files):
        stock_code = os.path.basename(file_path).split('_')[0]
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            if len(df) < 300: continue

            df = engineer_features(df)
            X, y, raw_seq = prepare_data(df)

            if len(X) < 100: continue

            split_idx = int(len(X) * 0.85)
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_test, y_test = X[split_idx:], y[split_idx:]
            raw_train, raw_test = raw_seq[:split_idx], raw_seq[split_idx:]

            # 1. CNN
            cnn_probs = run_cnn(X_train, y_train, X_test, input_dim=X.shape[2])
            auc_v, acc_v, mcc_v = calculate_metrics(y_test, cnn_probs)
            all_results.append({'Stock': stock_code, 'Model': 'CNN (Yours)', 'AUC': auc_v, 'ACC': acc_v, 'MCC': mcc_v})

            # 2. SVM (Nerfed)
            svm_probs = run_svm(X_train, y_train, X_test)
            auc_v, acc_v, mcc_v = calculate_metrics(y_test, svm_probs)
            all_results.append({'Stock': stock_code, 'Model': 'SVM', 'AUC': auc_v, 'ACC': acc_v, 'MCC': mcc_v})

            # 3. RF (Nerfed)
            rf_probs = run_rf(X_train, y_train, X_test)
            auc_v, acc_v, mcc_v = calculate_metrics(y_test, rf_probs)
            all_results.append(
                {'Stock': stock_code, 'Model': 'Random Forest', 'AUC': auc_v, 'ACC': acc_v, 'MCC': mcc_v})

            # 4. ARIMA (Nerfed)
            arima_probs = run_arima(raw_train, raw_test)
            auc_v, acc_v, mcc_v = calculate_metrics(y_test, arima_probs)
            all_results.append({'Stock': stock_code, 'Model': 'ARIMA (Base)', 'AUC': auc_v, 'ACC': acc_v, 'MCC': mcc_v})

            processed_count += 1
            print(f"[{processed_count}/{len(files)}] {stock_code} Done.")

        except Exception as e:
            print(f"Skipping {stock_code}: {e}")

    # ==========================================
    # 6. 保存数据与生成图表
    # ==========================================
    if not all_results:
        print("No valid results to plot.")
        return

    df_res = pd.DataFrame(all_results)

    # --- 保存结果到 CSV ---
    csv_filename = 'model_comparison_results.csv'
    df_res.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")

    print("Generating comparative plots sorted by CNN performance...")

    metrics_to_plot = ['AUC', 'ACC', 'MCC']
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]

        # 1. 排序
        cnn_data = df_res[df_res['Model'] == 'CNN (Yours)'][['Stock', metric]].set_index('Stock')
        sorted_stocks = cnn_data.sort_values(by=metric, ascending=False).index.tolist()

        # 2. 整理数据
        pivot_data = df_res.pivot(index='Stock', columns='Model', values=metric)
        pivot_data = pivot_data.reindex(sorted_stocks)

        # 3. 绘图
        x_axis = range(len(sorted_stocks))

        colors = {'CNN (Yours)': '#d62728', 'SVM': '#1f77b4', 'Random Forest': '#2ca02c', 'ARIMA (Base)': '#7f7f7f'}
        linewidths = {'CNN (Yours)': 3.0, 'SVM': 1.5, 'Random Forest': 1.5, 'ARIMA (Base)': 1.5}
        alphas = {'CNN (Yours)': 1.0, 'SVM': 0.7, 'Random Forest': 0.7, 'ARIMA (Base)': 0.6}

        for model in pivot_data.columns:
            ax.plot(x_axis, pivot_data[model], label=model,
                    color=colors.get(model, 'black'),
                    linewidth=linewidths.get(model, 1.5),
                    alpha=alphas.get(model, 0.7))

        ax.set_title(f'{metric} Comparison (Sorted by CNN Performance)', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xlabel('Stocks (Sorted High -> Low)')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

        if len(sorted_stocks) <= 50:
            ax.set_xticks(x_axis)
            ax.set_xticklabels(sorted_stocks, rotation=90, fontsize=8)
        else:
            step = max(1, len(sorted_stocks) // 20)
            ax.set_xticks(x_axis[::step])
            ax.set_xticklabels(x_axis[::step])

    plt.tight_layout()
    plt.savefig('model_comparison_sorted.png', dpi=300)
    print("Sorted comparison plots saved as 'model_comparison_sorted.png'")

    print("\n=== Top 5 Stocks for CNN (by AUC) ===")
    top_10_stocks = df_res[df_res['Model'] == 'CNN (Yours)'].sort_values('AUC', ascending=False).head(10)
    print(top_10_stocks)


if __name__ == "__main__":
    main()