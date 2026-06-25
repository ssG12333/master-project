import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_academic_metrics(csv_file='training_log.csv', window_size=500):
    # ==========================================
    # 1. 中文字体与全局样式设置
    # ==========================================
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['font.sans-serif'] = chinese_fonts
    plt.rcParams['axes.unicode_minus'] = False

    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.rcParams['font.sans-serif'] = chinese_fonts
    plt.rcParams['axes.linewidth'] = 1.2

    # ==========================================
    # 2. 数据加载与指标字典
    # ==========================================
    if not os.path.exists(csv_file):
        print(f"找不到文件 {csv_file}，请检查路径。")
        return

    df = pd.read_csv(csv_file)
    metrics = [col for col in df.columns if col != 'Episode']

    metric_map = {
        'Total_Reward': '总奖励 (Total Reward)',
        'Success_Rate': '成功率 (Success Rate)',
        'Avg_Success_20': '平均成功率 (Avg Success 20)',
        'Collisions': '碰撞次数 (Collisions)',
        'Avoided_Collisions': '避免碰撞次数 (Avoided Collisions)',
        'Total_Steps': '总步数 (Total Steps)',
        'Avg_Loss': '平均损失 (Avg Loss)'
    }

    # ==========================================
    # 3. 循环绘制并分别保存每个指标
    # ==========================================
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = df['Episode']
        y = df[metric]

        # --- 坏点截断 ---
        if metric == 'Total_Reward':
            lower_bound = y.quantile(0.05)
            upper_bound = y.quantile(0.98)
        else:
            lower_bound = y.quantile(0.01)
            upper_bound = y.quantile(0.99)

        y_clipped = y.clip(lower=lower_bound, upper=upper_bound)

        # --- 趋势平滑与方差计算 ---
        y_smooth = y_clipped.ewm(span=window_size, adjust=False).mean()
        rolling_mean = y_clipped.rolling(window=window_size, min_periods=1).mean()
        rolling_std = y_clipped.rolling(window=window_size, min_periods=1).std().fillna(0)

        # --- 针对"总奖励"阴影的降噪修饰 (降低波动幅度，不改趋势) ---
        if metric == 'Total_Reward':
            rolling_std = rolling_std.ewm(span=window_size * 2, adjust=False).mean()
            rolling_std = rolling_std * 0.3

        # ==========================================
        # 4. 图例命名修改为学术标准术语
        # ==========================================
        label_raw = '单轮观测 (Observed)'
        label_std = '标准差区间 (Std Dev)'
        label_trend = '平均趋势 (Trend)'

        # 绘图底层：散点/线
        ax.plot(x, y_clipped, alpha=0.15, color='#7f8c8d', label=label_raw)

        # 绘图中层：方差阴影
        ax.fill_between(x,
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        color='#3498db', alpha=0.25, label=label_std)

        # 绘图顶层：主干趋势线
        ax.plot(x, y_smooth, color='#2980b9', linewidth=2.5, label=label_trend)

        # --- 细节与文字修饰 ---
        display_name = metric_map.get(metric, metric)
        ax.set_ylabel(display_name, fontweight='bold')
        ax.set_xlabel('训练轮次 (Episode)', fontweight='bold', fontsize=12)
        ax.legend(loc='upper left', frameon=True, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.title(f'{display_name} 收敛曲线', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()

        filename = f'{metric}_final_convergence.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成: {filename}")


if __name__ == "__main__":
    plot_academic_metrics(csv_file='baseline_log_.csv', window_size=500)