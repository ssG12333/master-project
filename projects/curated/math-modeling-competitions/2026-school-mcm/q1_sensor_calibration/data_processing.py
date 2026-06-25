import pandas as pd
import numpy as np
from PyEMD import CEEMDAN
import matplotlib.pyplot as plt
import seaborn as sns

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
    
    def load_data(self):
        """读取Excel数据并处理时间戳"""
        print(f"[数据加载] 正在读取: {self.filepath}")
        df = pd.read_excel(self.filepath)
        
        # 自动识别列名
        df.columns = ['Time', 'A', 'B']
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
        
        # 计算相对时间（分钟）
        time_diff = df['Time'] - df['Time'].iloc[0]
        df['T_minutes'] = time_diff.dt.total_seconds() / 60.0
        
        print(f"[数据加载] 成功! 数据形状: {df.shape}")
        print(f"[数据加载] 时间范围: {df['Time'].min()} 至 {df['Time'].max()}")
        print(f"[数据加载] 数据A范围: [{df['A'].min():.3f}, {df['A'].max():.3f}]")
        print(f"[数据加载] 数据B范围: [{df['B'].min():.3f}, {df['B'].max():.3f}]")
        
        self.data = df
        return df
    
    def plot_eda(self, save_dir="eda_plots"):
        """生成原始数据探索性分析图表"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        df = self.data
        print(f"\n[EDA分析] 开始生成探索性分析图表...")
        
        # 图1: 原始数据A与B对比
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['Time'], df['A'], label='光纤传感器A', alpha=0.7, linewidth=0.8)
        ax.plot(df['Time'], df['B'], label='振弦传感器B（基准）', alpha=0.7, linewidth=0.8)
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('位移 (mm)', fontsize=12)
        ax.set_title('原始传感器数据对比: A vs B', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path1 = f"{save_dir}/01_raw_comparison.png"
        plt.savefig(path1, dpi=300)
        plt.close()
        print(f"  [EDA] 已保存: {path1}")
        
        # 图2: 误差随时间变化
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        error = df['A'] - df['B']
        axes[0].plot(df['Time'], error, color='coral', linewidth=0.8)
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[0].set_ylabel('绝对误差 A-B (mm)', fontsize=11)
        axes[0].set_title('传感器A与B的误差时序', fontsize=13)
        axes[0].grid(True, alpha=0.3)
        
        # 累积误差
        cum_error = np.abs(error).cumsum()
        axes[1].plot(df['Time'], cum_error, color='darkred', linewidth=1.2)
        axes[1].set_xlabel('时间', fontsize=12)
        axes[1].set_ylabel('累积误差 (mm)', fontsize=11)
        axes[1].set_title('累积误差增长趋势（体现时变漂移）', fontsize=13)
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        path2 = f"{save_dir}/02_error_analysis.png"
        plt.savefig(path2, dpi=300)
        plt.close()
        print(f"  [EDA] 已保存: {path2}")
        
        # 图3: A与B的散点关系
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # 全量散点（采样避免过密）
        sample_idx = np.linspace(0, len(df)-1, 2000, dtype=int)
        axes[0].scatter(df['A'].iloc[sample_idx], df['B'].iloc[sample_idx], 
                       alpha=0.4, s=5, c=range(len(sample_idx)), cmap='viridis')
        axes[0].plot([df['A'].min(), df['A'].max()], 
                    [df['A'].min(), df['A'].max()], 'r--', alpha=0.5, label='y=x')
        axes[0].set_xlabel('数据A (mm)', fontsize=11)
        axes[0].set_ylabel('数据B (mm)', fontsize=11)
        axes[0].set_title('A vs B 散点图（颜色=时间顺序）', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 误差分布
        axes[1].hist(error, bins=100, color='steelblue', alpha=0.7, edgecolor='white')
        axes[1].axvline(x=error.mean(), color='red', linestyle='--', label=f'均值={error.mean():.2f}')
        axes[1].set_xlabel('误差 A-B (mm)', fontsize=11)
        axes[1].set_ylabel('频次', fontsize=11)
        axes[1].set_title('误差分布直方图', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        path3 = f"{save_dir}/03_scatter_distribution.png"
        plt.savefig(path3, dpi=300)
        plt.close()
        print(f"  [EDA] 已保存: {path3}")
        
        # 图4: 数据统计摘要表
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        stats_text = f"""数据统计摘要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
数据A (光纤传感器):
  均值: {df['A'].mean():.3f} mm
  标准差: {df['A'].std():.3f} mm
  最小值: {df['A'].min():.3f} mm
  最大值: {df['A'].max():.3f} mm

数据B (振弦传感器):
  均值: {df['B'].mean():.3f} mm
  标准差: {df['B'].std():.3f} mm
  最小值: {df['B'].min():.3f} mm
  最大值: {df['B'].max():.3f} mm

误差 (A - B):
  均值: {error.mean():.3f} mm
  标准差: {error.std():.3f} mm
  最大偏差: {error.abs().max():.3f} mm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
相关系数 r = {df['A'].corr(df['B']):.6f}
"""
        ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.tight_layout()
        path4 = f"{save_dir}/04_statistics_summary.png"
        plt.savefig(path4, dpi=300)
        plt.close()
        print(f"  [EDA] 已保存: {path4}")
        
        print(f"[EDA分析] 完成! 共生成4张图表于 {save_dir}/ 目录\n")
    
    def apply_ceemdan(self, a_series, max_imf=5):
        """对数据A进行CEEMDAN分解"""
        print(f"[CEEMDAN] 正在分解 {len(a_series)} 个数据点...")
        ceemdan = CEEMDAN()
        imfs = ceemdan(a_series.values, max_imf=max_imf)
        
        if len(imfs) > 1:
            a_smooth = np.sum(imfs[1:], axis=0)
        else:
            a_smooth = imfs[0]
            
        print(f"[CEEMDAN] 分解完成! 得到 {len(imfs)} 个IMF分量")
        return imfs, a_smooth
    
    def extract_features(self, df, a_smooth):
        """构建特征用于模型训练"""
        df = df.copy()
        df['A_smooth'] = a_smooth
        df['A_sq'] = df['A_smooth'] ** 2
        df['A_cube'] = df['A_smooth'] ** 3
        df['A_vel'] = df['A_smooth'].diff().fillna(0)
        df['A_bin'] = pd.qcut(df['A_smooth'], q=10, labels=False, duplicates='drop')
        
        df['A_diff2'] = df['A_smooth'].diff(2).fillna(0)
        df['A_ratio'] = df['A_smooth'] / (df['A_smooth'].rolling(window=50).mean().fillna(df['A_smooth']))
        df['A_rolling_mean_50'] = df['A_smooth'].rolling(window=50, min_periods=1).mean()
        df['A_rolling_std_50'] = df['A_smooth'].rolling(window=50, min_periods=1).std().fillna(0)
        df['A_deviation'] = df['A_smooth'] - df['A_rolling_mean_50']
        df['T_norm'] = df['T_minutes'] / df['T_minutes'].max()
        
        return df
    
    def get_prediction_features(self, target_a_values, df_train):
        """对5个孤立值构建预测特征"""
        pred_df = pd.DataFrame({'A': target_a_values})
        return pred_df


if __name__ == "__main__":
    print("="*50)
    print("测试数据处理模块")
    print("="*50)
