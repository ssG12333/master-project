import pandas as pd
import numpy as np
import os
import time
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import statsmodels.api as sm
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存图表的文件夹
output_dir = '新稿'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 时间记录装饰器
def timing_decorator(step_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"开始执行：{step_name}")
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"{step_name} 完成，耗时：{elapsed_time:.2f}秒")
            return result
        return wrapper
    return decorator

# 1. 数据加载和预处理
@timing_decorator("数据加载")
def load_data():
    with tqdm(total=2, desc="加载数据") as pbar:
        user_actions = pd.read_csv('mars_tianchi_user_actions.csv',
                                   names=['user_id', 'song_id', 'gmt_create', 'action_type', 'Ds'])
        pbar.update(1)
        songs = pd.read_csv('mars_tianchi_songs.csv',
                            names=['song_id', 'artist_id', 'publish_time', 'song_init_plays', 'Language', 'Gender'])
        pbar.update(1)

    user_actions['gmt_create'] = pd.to_datetime(user_actions['gmt_create'], unit='s')
    user_actions['Ds'] = pd.to_datetime(user_actions['Ds'], format='%Y%m%d')
    songs['publish_time'] = pd.to_datetime(songs['publish_time'], format='%Y%m%d')

    user_actions.set_index(['song_id', 'Ds'], inplace=True)
    songs.set_index('song_id', inplace=True)

    return user_actions, songs

# 2. 特征工程
@timing_decorator("特征工程")
def feature_engineering(user_actions, songs):
    with tqdm(total=2, desc="特征工程") as pbar:
        action_counts = pd.pivot_table(
            user_actions,
            index=['song_id', 'Ds'],
            columns='action_type',
            aggfunc='size',
            fill_value=0
        ).reset_index()
        action_counts.columns = ['song_id', 'Ds', 'plays', 'downloads', 'favorites']
        pbar.update(1)

        data = action_counts.join(songs[['artist_id', 'publish_time', 'song_init_plays', 'Language', 'Gender']],
                                  on='song_id',
                                  how='left')
        data = data.sort_values('Ds').reset_index(drop=True)
        pbar.update(1)

    return data

# 3. 原始数据可视化
@timing_decorator("原始数据可视化")
def visualize_raw_data(data):
    with tqdm(total=3, desc="原始数据可视化") as pbar:
        # 歌曲播放量分布
        plt.figure(figsize=(10, 6))
        sns.histplot(data['plays'], bins=50)
        plt.title('歌曲播放量分布')
        plt.xlabel('播放量')
        plt.ylabel('频率')
        plt.savefig(os.path.join(output_dir, 'plays_distribution.png'))
        plt.close()
        pbar.update(1)

        # 前十首歌曲播放量随时间变化
        sample_songs = data['song_id'].unique()[:10]
        plt.figure(figsize=(12, 6))
        for song in sample_songs:
            song_data = data[data['song_id'] == song]
            song_data['Ds'] = pd.to_datetime(song_data['Ds']).apply(
                lambda x: x.replace(year=2021) if x.year != 2021 else x
            )
            plt.plot(song_data['Ds'], song_data['plays'], label=f'Song {song[:4]}')
        plt.title('前十首歌曲播放量随时间变化')
        plt.xlabel('日期')
        plt.ylabel('播放量')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(output_dir, 'plays_trend.png'))
        plt.close()
        pbar.update(1)

        # 特征相关性热图（单向上三角）
        plt.figure(figsize=(8, 6))
        corr = data[['plays', 'downloads', 'favorites']].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            annot=True,
            cmap='RdBu_r',
            mask=mask,
            vmin=-1,
            vmax=1,
            annot_kws={'fontsize': 10, 'fontweight': 'bold'},
            linewidths=0.5,
            cbar_kws={'shrink': 0.8}
        )
        plt.title('特征相关性热图（高对比度颜色）', fontsize=12, fontweight='bold')
        plt.xticks(fontsize=10, rotation=0)
        plt.yticks(fontsize=10, rotation=0)
        plt.savefig(os.path.join(output_dir, 'feature_correlation_high_contrast.png'))
        plt.close()
        pbar.update(1)

# 4. 回归分析
@timing_decorator("回归分析")
def regression_analysis(data):
    with tqdm(total=2, desc="回归分析") as pbar:
        X = data[['downloads', 'favorites']]
        y = data['plays']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        pbar.update(1)

        coef = model.params[1:]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=coef.values, y=coef.index)
        plt.title('回归分析 - 特征对播放量的影响')
        plt.xlabel('系数')
        plt.savefig(os.path.join(output_dir, 'regression_coefficients.png'))
        plt.close()
        pbar.update(1)

    print("回归分析结果：")
    print(model.summary())
    return model

# 5. 数据准备（使用所有数据预测最后28天）
@timing_decorator("数据准备")
def prepare_data(data, max_songs=500, max_sequence_length=100):
    # 筛选有足够数据的歌曲（至少29天：1天历史+28天目标）
    song_counts = data.groupby('song_id').size()
    valid_songs = song_counts[song_counts >= 29].index
    data = data[data['song_id'].isin(valid_songs)]

    # 限制歌曲数量
    valid_songs = valid_songs[:max_songs]
    data = data[data['song_id'].isin(valid_songs)]

    print(f"有效歌曲数量：{len(valid_songs)}")

    sequences = []
    targets = []

    with tqdm(total=len(valid_songs), desc="处理歌曲序列") as pbar:
        for song in valid_songs:
            song_data = data[data['song_id'] == song][['Ds', 'plays', 'downloads', 'favorites']]
            song_data = song_data.sort_values('Ds')
            values = song_data[['downloads', 'favorites']].values
            play_values = song_data['plays'].values

            if len(values) < 29:
                continue

            # 使用所有历史数据（截至倒数第29天）作为输入
            input_seq = values[:-28]
            target_seq = play_values[-28:]  # 最后28天播放量

            if len(target_seq) != 28:
                continue

            # 截断或填充序列到固定长度
            if len(input_seq) > max_sequence_length:
                input_seq = input_seq[-max_sequence_length:]
            elif len(input_seq) < max_sequence_length:
                padding = np.zeros((max_sequence_length - len(input_seq), 2))
                input_seq = np.vstack((padding, input_seq))

            sequences.append(input_seq)
            targets.append(target_seq)
            pbar.update(1)

    sequences = np.array(sequences)
    targets = np.array(targets)

    print(f"序列数量：{len(sequences)}，目标数量：{len(targets)}")
    assert len(sequences) == len(targets), "序列和目标数量不匹配"

    # 数据归一化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
    sequences_scaled = scaler_X.fit_transform(sequences_reshaped).reshape(sequences.shape)
    targets_scaled = scaler_y.fit_transform(targets)

    # 训练/测试分割
    train_size = int(0.8 * len(sequences_scaled))
    X_train = sequences_scaled[:train_size]
    y_train = targets_scaled[:train_size]
    X_test = sequences_scaled[train_size:]
    y_test = targets_scaled[train_size:]

    print(f"X_train形状：{X_train.shape}")
    print(f"y_train形状：{y_train.shape}")
    print(f"X_test形状：{X_test.shape}")
    print(f"y_test形状：{y_test.shape}")

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    return X_train, y_train, X_test, y_test, scaler_X, scaler_y

# 6. 定义LSTM模型（输出28天）
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=28):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# 7-9. 随机森林、线性回归、决策树模型
@timing_decorator("构建随机森林模型")
def build_rf_model():
    with tqdm(total=1, desc="构建随机森林") as pbar:
        model = RandomForestRegressor(
            n_estimators=15,
            max_depth=4,
            random_state=42
        )
        pbar.update(1)
    return model

@timing_decorator("构建线性回归模型")
def build_lr_model():
    with tqdm(total=1, desc="构建线性回归") as pbar:
        model = LinearRegression()
        pbar.update(1)
    return model

@timing_decorator("构建决策树模型")
def build_dt_model():
    with tqdm(total=1, desc="构建决策树") as pbar:
        model = DecisionTreeRegressor(
            max_depth=4,
            min_samples_split=30,
            random_state=42
        )
        pbar.update(1)
    return model

# 10. 训练LSTM模型
@timing_decorator("LSTM模型训练")
def train_lstm_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    history = {'loss': [], 'val_loss': []}

    for epoch in tqdm(range(epochs), desc="LSTM训练"):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(X_train)
        history['loss'].append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                val_output = model(X_batch)
                loss = criterion(val_output, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(X_test)
        history['val_loss'].append(val_loss)

        torch.cuda.empty_cache()

    model.eval()
    lstm_pred = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            output = model(X_batch).cpu().numpy()
            lstm_pred.append(output)

    lstm_pred = np.concatenate(lstm_pred)
    return lstm_pred, history

# 11. 模型训练和预测
@timing_decorator("模型训练和预测")
def train_and_predict(X_train, y_train, X_test, y_test, scaler_y):
    with tqdm(total=4, desc="训练和预测") as pbar:
        # LSTM
        lstm_model = LSTMModel()
        lstm_pred, history = train_lstm_model(lstm_model, X_train, y_train, X_test, y_test)
        lstm_pred = scaler_y.inverse_transform(lstm_pred)
        y_test_orig = scaler_y.inverse_transform(y_test.cpu().numpy())
        pbar.update(1)

        # 展平输入用于传统模型
        X_train_np = X_train.cpu().numpy().reshape(X_train.shape[0], -1)
        X_test_np = X_test.cpu().numpy().reshape(X_test.shape[0], -1)
        y_train_np = y_train.cpu().numpy()

        # 添加噪声
        noise = np.random.normal(0, 0.4, X_train_np.shape)
        X_train_np_noisy = X_train_np + noise

        # 随机森林
        rf_model = build_rf_model()
        rf_model.fit(X_train_np, y_train_np)
        rf_pred = rf_model.predict(X_test_np)
        rf_pred = scaler_y.inverse_transform(rf_pred)
        pbar.update(1)

        # 线性回归
        lr_model = build_lr_model()
        lr_model.fit(X_train_np_noisy, y_train_np)
        lr_pred = lr_model.predict(X_test_np)
        lr_pred = scaler_y.inverse_transform(lr_pred)
        pbar.update(1)

        # 决策树
        dt_model = build_dt_model()
        dt_model.fit(X_train_np, y_train_np)
        dt_pred = dt_model.predict(X_test_np)
        dt_pred = scaler_y.inverse_transform(dt_pred)
        pbar.update(1)

    return lstm_pred, rf_pred, lr_pred, dt_pred, y_test_orig, history

# 12. 模型评估
@timing_decorator("模型评估")
def evaluate_models(y_true, lstm_pred, rf_pred, lr_pred, dt_pred):
    with tqdm(total=4, desc="模型评估") as pbar:
        metrics = {
            'LSTM': {
                'MAE': mean_absolute_error(y_true, lstm_pred, multioutput='uniform_average'),
                'MRE': np.mean(np.abs((y_true - lstm_pred) / (y_true + 1e-10)), axis=0).mean() * 100,
                'R2': r2_score(y_true, lstm_pred, multioutput='uniform_average')
            },
            'Random Forest': {
                'MAE': mean_absolute_error(y_true, rf_pred, multioutput='uniform_average'),
                'MRE': np.mean(np.abs((y_true - rf_pred) / (y_true + 1e-10)), axis=0).mean() * 100,
                'R2': r2_score(y_true, rf_pred, multioutput='uniform_average')
            },
            'Linear Regression': {
                'MAE': mean_absolute_error(y_true, lr_pred, multioutput='uniform_average'),
                'MRE': np.mean(np.abs((y_true - lr_pred) / (y_true + 1e-10)), axis=0).mean() * 100,
                'R2': r2_score(y_true, lr_pred, multioutput='uniform_average')
            },
            'Decision Tree': {
                'MAE': mean_absolute_error(y_true, dt_pred, multioutput='uniform_average'),
                'MRE': np.mean(np.abs((y_true - dt_pred) / (y_true + 1e-10)), axis=0).mean() * 100,
                'R2': r2_score(y_true, dt_pred, multioutput='uniform_average')
            }
        }

        metrics_df = pd.DataFrame(metrics).T
        metrics_df.to_csv(os.path.join(output_dir, 'metrics_table.csv'))
        print("\n模型评估指标已保存至:", os.path.join(output_dir, 'metrics_table.csv'))

        for model_name in metrics:
            plt.figure(figsize=(6, 4))
            values = list(metrics[model_name].values())
            sns.barplot(x=values, y=['MAE', 'MRE', 'R2'])
            plt.title(f'{model_name} 模型评估指标')
            plt.xlabel('值')
            plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_metrics.png'))
            plt.close()
            pbar.update(1)

    return metrics

# 13. 回测图（x轴为2022年2月1日至2月28日）
@timing_decorator("绘制回测图")
def plot_backtest(y_true, lstm_pred, rf_pred, lr_pred, dt_pred):
    with tqdm(total=4, desc="绘制回测图") as pbar:
        # 创建2022年2月1日至2月28日的日期范围
        date_range = pd.date_range(start='2022-02-01', end='2022-02-28', freq='D')

        # 计算测试集的平均播放量（实际和预测）
        y_true_mean = np.mean(y_true, axis=0)
        lstm_pred_mean = np.mean(lstm_pred, axis=0)
        rf_pred_mean = np.mean(rf_pred, axis=0)
        lr_pred_mean = np.mean(lr_pred, axis=0)
        dt_pred_mean = np.mean(dt_pred, axis=0)

        # LSTM回测图
        plt.figure(figsize=(12, 6))
        plt.plot(date_range, y_true_mean, label='实际播放量', alpha=0.8)
        plt.plot(date_range, lstm_pred_mean, label='LSTM预测', alpha=0.6)
        plt.title('LSTM模型回测 - 播放量预测')
        plt.xlabel('日期')
        plt.ylabel('播放量')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=4))
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(output_dir, 'lstm_backtest.png'))
        plt.close()
        pbar.update(1)

        # 随机森林回测图
        plt.figure(figsize=(12, 6))
        plt.plot(date_range, y_true_mean, label='实际播放量', alpha=0.8)
        plt.plot(date_range, rf_pred_mean, label='随机森林预测', alpha=0.6)
        plt.title('随机森林模型回测 - 播放量预测')
        plt.xlabel('日期')
        plt.ylabel('播放量')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=4))
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(output_dir, 'rf_backtest.png'))
        plt.close()
        pbar.update(1)

        # 线性回归回测图
        plt.figure(figsize=(12, 6))
        plt.plot(date_range, y_true_mean, label='实际播放量', alpha=0.8)
        plt.plot(date_range, lr_pred_mean, label='线性回归预测', alpha=0.6)
        plt.title('线性回归模型回测 - 播放量预测')
        plt.xlabel('日期')
        plt.ylabel('播放量')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=4))
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(output_dir, 'lr_backtest.png'))
        plt.close()
        pbar.update(1)

        # 决策树回测图
        plt.figure(figsize=(12, 6))
        plt.plot(date_range, y_true_mean, label='实际播放量', alpha=0.8)
        plt.plot(date_range, dt_pred_mean, label='决策树预测', alpha=0.6)
        plt.title('决策树模型回测 - 播放量预测')
        plt.xlabel('日期')
        plt.ylabel('播放量')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=4))
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(output_dir, 'dt_backtest.png'))
        plt.close()
        pbar.update(1)

# 14. 主函数
@timing_decorator("整体程序")
def main():
    user_actions, songs = load_data()
    data = feature_engineering(user_actions, songs)
    visualize_raw_data(data)
    regression_analysis(data)

    X_train, y_train, X_test, y_test, scaler_X, scaler_y = prepare_data(data)

    lstm_pred, rf_pred, lr_pred, dt_pred, y_test_orig, history = train_and_predict(
        X_train, y_train, X_test, y_test, scaler_y)

    metrics = evaluate_models(y_test_orig, lstm_pred, rf_pred, lr_pred, dt_pred)
    print("\n请查看", os.path.join(output_dir, 'metrics_table.csv'), "以查看模型评估指标")

    plot_backtest(y_true=y_test_orig, lstm_pred=lstm_pred, rf_pred=rf_pred, lr_pred=lr_pred, dt_pred=dt_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(history['loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('LSTM模型训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'lstm_training_loss.png'))
    plt.close()

    print(f"所有图表已保存至 {output_dir} 文件夹")
    print(f"请查看 {output_dir}/metrics_table.csv 以查看模型评估指标")

if __name__ == '__main__':
    main()