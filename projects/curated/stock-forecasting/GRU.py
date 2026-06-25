import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm


# --- 1. 双分支模型设计 ---
class TwoBranchGRU(nn.Module):
    def __init__(self, seq_dim, factor_dim, hidden_dim=64):
        super(TwoBranchGRU, self).__init__()

        # 分支 1: GRU 处理基础价格序列 (OHLCV)
        self.gru_branch = nn.GRU(seq_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)

        # 分支 2: MLP 处理技术因子截面 (Technical Factors)
        self.factor_branch = nn.Sequential(
            nn.Linear(factor_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 融合层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, seq_x, factor_x):
        # seq_x: (batch, window, seq_dim)
        # factor_x: (batch, factor_dim)

        _, h_n = self.gru_branch(seq_x)
        gru_out = h_n[-1]  # 取最后一层最后一个隐状态

        factor_out = self.factor_branch(factor_x)

        combined = torch.cat((gru_out, factor_out), dim=1)
        return self.fc(combined).squeeze(-1)


# --- 2. 滚动训练核心逻辑 ---
def train_rolling_model(df, window=30, train_size=252, step=1):
    """
    针对单只股票进行滚动训练
    df: 已经过预处理的单股数据
    window: GRU 观察的时间步 (T=30)
    train_size: 滚动训练的窗口大小 (例如用过去一年的数据训练)
    """
    # 区分基础数据特征和技术因子特征
    base_cols = ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm']
    factor_cols = [c for c in df.columns if c.endswith('_norm') and c not in base_cols]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoBranchGRU(len(base_cols), len(factor_cols)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    predictions = []

    # 准备时序数据块
    data_points = []
    for i in range(len(df) - window):
        seq = df[base_cols].iloc[i: i + window].values
        factor = df[factor_cols].iloc[i + window - 1].values  # 取当前时刻的因子
        label = df['target_return'].iloc[i + window - 1]
        date = df['date'].iloc[i + window - 1]
        data_points.append({'seq': seq, 'factor': factor, 'label': label, 'date': date})

    # 开始滚动预测 (从 train_size 开始，预测之后的一天)
    # 注意：为了提速，实际操作中通常每隔 N 天更新一次权重，这里展示逐日更新逻辑
    for t in range(train_size, len(data_points), step):
        # 训练集：[t - train_size, t)
        train_batch = data_points[t - train_size: t]
        # 测试集（预测目标）：[t]
        test_point = data_points[t]

        # 转换为 Tensor
        train_seq = torch.FloatTensor(np.array([p['seq'] for p in train_batch])).to(device)
        train_fac = torch.FloatTensor(np.array([p['factor'] for p in train_batch])).to(device)
        train_y = torch.FloatTensor(np.array([p['label'] for p in train_batch])).to(device)

        # 快速训练 (Fine-tuning)
        model.train()
        optimizer.zero_grad()
        out = model(train_seq, train_fac)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()

        # 预测
        model.eval()
        with torch.no_grad():
            test_seq = torch.FloatTensor(test_point['seq']).unsqueeze(0).to(device)
            test_fac = torch.FloatTensor(test_point['factor']).unsqueeze(0).to(device)
            pred = model(test_seq, test_fac).item()

            predictions.append({
                'date': test_point['date'],
                'actual_return': test_point['label'],
                'pred_return': pred
            })

    return pd.DataFrame(predictions)


def main():
    data_dir = 'data_processed'
    files = glob.glob(os.path.join(data_dir, "*_processed.csv"))

    all_results = []

    print(f"开始逐股滚动训练，共 {len(files)} 只股票...")
    for f in files:
        stock_code = os.path.basename(f).split('_')[0]
        df_stock = pd.read_csv(f)

        # 执行滚动预测
        print(f"正在处理股票: {stock_code}")
        res = train_rolling_model(df_stock)

        if res is not None:
            res['code'] = stock_code
            all_results.append(res)
            # 每处理完一只股票保存一次，防止崩溃
            res.to_csv(f"pred_{stock_code}.csv", index=False)

    # 汇总
    final_df = pd.concat(all_results)
    final_df.to_csv("gru_predictions.csv", index=False)
    print("全部股票滚动预测完成。")


if __name__ == "__main__":
    main()