import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Q1Visualizer:
    @staticmethod
    def plot_ceemdan(time_series, original_a, imfs, a_smooth, save_path="ceemdan_spectrogram.png"):
        """图1：CEEMDAN 模态分解时频图"""
        num_imfs = min(len(imfs), 4)
        fig, axes = plt.subplots(num_imfs + 2, 1, figsize=(14, 10), sharex=True)
        
        axes[0].plot(time_series, original_a, color='black', linewidth=1)
        axes[0].set_title("原始光纤数据 A (含高频跳变)", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("原始A (mm)")
        axes[0].grid(True, alpha=0.3)
        
        for i in range(num_imfs):
            color = 'red' if i == 0 else 'royalblue'
            axes[i+1].plot(time_series, imfs[i], color=color, linewidth=0.8)
            axes[i+1].set_ylabel(f"IMF {i+1}", fontsize=10)
            if i == 0:
                axes[i+1].set_title("IMF1 - 高频设备毛刺模态 (将被滤除)", color='red', fontsize=11)
            else:
                axes[i+1].set_title(f"IMF{i+1} - 局部应力波动")
            axes[i+1].grid(True, alpha=0.3)
            
        axes[-1].plot(time_series, a_smooth, color='green', linewidth=1.5)
        axes[-1].set_title("剔除高频后的平滑趋势重构 (A_smooth)", fontsize=13, fontweight='bold')
        axes[-1].set_xlabel("时间", fontsize=12)
        axes[-1].set_ylabel("A_smooth (mm)")
        axes[-1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_drift_main(time_series, A, B, A_calibrated, save_path="02_drift_main.png"):
        """图2-1：漂移校正主图（独立）"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        raw_error = np.abs(A - B)
        cal_error = np.abs(A_calibrated - B)
        
        ax.fill_between(time_series, 0, raw_error, 
                       color='red', alpha=0.08, zorder=1)
        ax.fill_between(time_series, B - cal_error, B + cal_error,
                       color='dodgerblue', alpha=0.06, zorder=2)
        
        ax.plot(time_series, A, label='原始光纤数据A (校正前)', 
               color='gray', alpha=0.5, linestyle='--', linewidth=1.2, zorder=3)
        ax.plot(time_series, A_calibrated, label='Ridge校正后数据A\'', 
               color='dodgerblue', linewidth=2, zorder=4)
        ax.plot(time_series, B, label='基准数据B (振弦式真值)', 
               color='black', linewidth=2.5, zorder=5)
        
        ax.set_xlabel('时间索引', fontsize=12, fontweight='bold')
        ax.set_ylabel('位移 (mm)', fontsize=12, fontweight='bold')
        ax.set_title('非线性时变漂移校正效果', fontsize=14, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_error_comparison(time_series, A, B, A_calibrated, save_path="02_error_comparison.png"):
        """图2-2：校正前后误差对比（独立）"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        raw_error = np.abs(A - B)
        cal_error = np.abs(A_calibrated - B)
        
        ax.fill_between(time_series, 0, raw_error, color='red', alpha=0.15, zorder=1)
        ax.fill_between(time_series, 0, cal_error, color='dodgerblue', alpha=0.1, zorder=2)
        ax.plot(time_series, raw_error, color='red', linewidth=1.5, alpha=0.7, label='校正前误差 |A-B|')
        ax.plot(time_series, cal_error, color='dodgerblue', linewidth=1.8, alpha=0.8, label='校正后误差 |A\'-B|')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax.set_xlabel('时间索引', fontsize=12, fontweight='bold')
        ax.set_ylabel('绝对误差 (mm)', fontsize=12, fontweight='bold')
        ax.set_title('校正前后绝对误差对比', fontsize=14, fontweight='bold', pad=10)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_drift_zoom(time_series, A, B, A_calibrated, save_path="02_drift_zoom.png"):
        """图2-3：局部放大校正细节（独立）"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        zoom_start = max(0, len(time_series) // 4)
        zoom_end = min(zoom_start + 1000, len(time_series))
        
        cal_error = np.abs(A_calibrated - B)
        
        ax.fill_between(time_series[zoom_start:zoom_end], 
                       B[zoom_start:zoom_end] - cal_error[zoom_start:zoom_end],
                       B[zoom_start:zoom_end] + cal_error[zoom_start:zoom_end],
                       color='dodgerblue', alpha=0.1, zorder=2)
        ax.plot(time_series[zoom_start:zoom_end], A[zoom_start:zoom_end], 
               color='gray', alpha=0.6, linestyle='--', linewidth=1.2, zorder=3)
        ax.plot(time_series[zoom_start:zoom_end], A_calibrated[zoom_start:zoom_end], 
               color='dodgerblue', linewidth=2.5, zorder=4)
        ax.plot(time_series[zoom_start:zoom_end], B[zoom_start:zoom_end], 
               color='black', linewidth=3, zorder=5)
        
        ax.set_xlabel('时间索引', fontsize=12, fontweight='bold')
        ax.set_ylabel('位移 (mm)', fontsize=12, fontweight='bold')
        ax.set_title(f'局部放大 (索引 {zoom_start}-{zoom_end}): Ridge校正细节对比', 
                    fontsize=14, fontweight='bold', pad=10)
        ax.legend(['原始A', '校正后A\'', '基准B'], loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_residual_heatmap(A_values, B_values, A_calibrated, save_path="03_residual_heatmap.png"):
        """图3：校正残差与校正量二维热力图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        calibration_amount = np.abs(A_values - A_calibrated)
        residual_error = np.abs(A_calibrated - B_values)
        
        hb1 = axes[0].hexbin(A_values, calibration_amount, gridsize=50, cmap='Reds', mincnt=1)
        axes[0].set_title('模型动态施加的绝对校正量分布\n|A - A\'|', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('原始光纤测量值 A (mm)', fontsize=11)
        axes[0].set_ylabel('校正力度 (mm)', fontsize=11)
        fig.colorbar(hb1, ax=axes[0], label='样本密度')
        
        hb2 = axes[1].hexbin(A_values, residual_error, gridsize=50, cmap='Blues', mincnt=1)
        axes[1].set_title('校正后残差分布\n理想状态紧贴 Y=0', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('原始光纤测量值 A (mm)', fontsize=11)
        axes[1].set_ylabel('残差 |A\' - B| (mm)', fontsize=11)
        fig.colorbar(hb2, ax=axes[1], label='样本密度')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_calibration_amount(A_values, A_calibrated, save_path="03_calibration_amount.png"):
        """图3-2：校正量分布直方图（独立）"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        calibration_amount = A_values - A_calibrated
        
        ax.hist(calibration_amount, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.axvline(x=np.mean(calibration_amount), color='blue', linestyle='-', linewidth=2, 
                  label=f'均值={np.mean(calibration_amount):.2f} mm')
        
        ax.set_xlabel('校正量 A - A\' (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('频次', fontsize=12, fontweight='bold')
        ax.set_title('Ridge模型校正量分布', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_model_comparison_original(time_series, A, B, save_path="04_original_comparison.png"):
        """图4-1：原始数据对比（独立）"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        step = max(1, len(time_series) // 2000)
        
        ax.plot(time_series[::step], A[::step], alpha=0.6, linewidth=1, color='gray', label='原始A')
        ax.plot(time_series[::step], B[::step], linewidth=2, color='black', label='基准B')
        
        ax.set_xlabel('时间索引', fontsize=12, fontweight='bold')
        ax.set_ylabel('位移 (mm)', fontsize=12, fontweight='bold')
        ax.set_title('原始数据对比: 光纤A vs 振弦式B', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_model_comparison_results(time_series, B, xgb_preds, poly_preds, best_preds, 
                                      best_model_name="最佳模型", save_path="04_model_comparison.png"):
        """图4-2：多模型校正效果对比（独立）"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        step = max(1, len(time_series) // 2000)
        
        ax.plot(time_series[::step], B[::step], linewidth=2.5, color='black', label='基准B (真值)', zorder=5)
        ax.plot(time_series[::step], xgb_preds[::step], linewidth=1.2, color='red', alpha=0.7, label='Ridge回归', zorder=3)
        ax.plot(time_series[::step], poly_preds[::step], linewidth=1.2, color='green', alpha=0.7, label='多项式回归(3阶)', zorder=3)
        ax.plot(time_series[::step], best_preds[::step], linewidth=1.5, color='dodgerblue', alpha=0.9, label=best_model_name, zorder=4)
        
        ax.set_xlabel('时间索引', fontsize=12, fontweight='bold')
        ax.set_ylabel('位移 (mm)', fontsize=12, fontweight='bold')
        ax.set_title('多模型校正效果对比', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_model_zoom(time_series, B, A, best_preds, best_model_name="最佳模型", save_path="04_zoom.png"):
        """图4-3：局部放大校正细节（独立）"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        zoom_end = min(800, len(time_series))
        
        ax.plot(time_series[:zoom_end], B[:zoom_end], linewidth=2.5, color='black', label='基准B (真值)', zorder=5)
        ax.plot(time_series[:zoom_end], A[:zoom_end], linewidth=1.2, color='gray', alpha=0.5, linestyle='--', label='原始A', zorder=1)
        ax.plot(time_series[:zoom_end], best_preds[:zoom_end], linewidth=2, color='dodgerblue', alpha=0.9, label=f'{best_model_name}校正后', zorder=4)
        
        ax.fill_between(time_series[:zoom_end], 
                       B[:zoom_end] - np.abs(best_preds[:zoom_end] - B[:zoom_end]),
                       B[:zoom_end] + np.abs(best_preds[:zoom_end] - B[:zoom_end]),
                       color='dodgerblue', alpha=0.08, zorder=3)
        
        ax.set_xlabel('时间索引', fontsize=12, fontweight='bold')
        ax.set_ylabel('位移 (mm)', fontsize=12, fontweight='bold')
        ax.set_title(f'局部放大 (前{zoom_end}点): {best_model_name}校正细节对比', 
                    fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_error_histogram(A, B, best_preds, best_model_name="最佳模型", save_path="05_error_histogram.png"):
        """图5-1：误差分布直方图（独立）"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        raw_error = A - B
        best_error = best_preds - B
        
        max_err = np.percentile(np.abs(raw_error), 99)
        bins = np.linspace(-max_err, max_err, 60)
        
        ax.hist(raw_error, bins=bins, alpha=0.5, color='gray', label='校正前 (原始A)', density=True)
        ax.hist(best_error, bins=bins, alpha=0.6, color='dodgerblue', label=f'{best_model_name}校正后', density=True)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('误差 (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('密度', fontsize=12, fontweight='bold')
        ax.set_title('校正前后误差分布对比', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_error_boxplot(A, B, best_preds, best_model_name="最佳模型", save_path="05_error_boxplot.png"):
        """图5-2：误差箱线图对比（独立）"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        raw_error = A - B
        best_error = best_preds - B
        
        error_data = [raw_error, best_error]
        bp = ax.boxplot(error_data, labels=['校正前 (原始A)', f'{best_model_name}\n校正后'], 
                       patch_artist=True, widths=0.5)
        colors = ['gray', 'dodgerblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('误差 (mm)', fontsize=12, fontweight='bold')
        ax.set_title('校正前后误差箱线图对比', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_error_cumulative(A, B, best_preds, best_model_name="最佳模型", save_path="05_error_cumulative.png"):
        """图5-3：累积误差增长曲线（独立）"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        raw_error = A - B
        best_error = best_preds - B
        
        cum_raw = np.cumsum(np.abs(raw_error))
        cum_best = np.cumsum(np.abs(best_error))
        
        ax.plot(cum_raw, color='gray', linewidth=2.5, label='校正前 (原始A)', alpha=0.8)
        ax.plot(cum_best, color='dodgerblue', linewidth=2.5, label=f'{best_model_name}校正后')
        ax.fill_between(range(len(cum_best)), cum_best, alpha=0.1, color='dodgerblue')
        
        ax.set_xlabel('时间步', fontsize=12, fontweight='bold')
        ax.set_ylabel('累积误差 (mm)', fontsize=12, fontweight='bold')
        ax.set_title('累积误差增长曲线', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_error_vs_A(A, B, best_preds, best_model_name="最佳模型", save_path="05_error_vs_A.png"):
        """图5-4：绝对误差 vs A值散点图（独立）"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        raw_error = A - B
        best_error = best_preds - B
        
        sample_size = min(5000, len(A))
        sample_idx = np.random.choice(len(A), sample_size, replace=False)
        
        ax.scatter(A[sample_idx], np.abs(raw_error[sample_idx]), 
                  alpha=0.3, s=8, color='gray', label='校正前', zorder=2)
        ax.scatter(A[sample_idx], np.abs(best_error[sample_idx]), 
                  alpha=0.4, s=8, color='dodgerblue', label=f'{best_model_name}校正后', zorder=3)
        
        ax.set_xlabel('A值 (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('|误差| (mm)', fontsize=12, fontweight='bold')
        ax.set_title('绝对误差 vs 原始测量值A', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_error_qq(best_error, best_model_name="最佳模型", save_path="05_error_qq.png"):
        """图5-5：误差正态QQ图（独立）"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        stats.probplot(best_error, dist="norm", plot=ax)
        
        ax.set_title(f'{best_model_name}校正后误差正态QQ图', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_error_timeseries(A, B, best_preds, best_model_name="最佳模型", save_path="05_error_timeseries.png"):
        """图5-6：绝对误差时序变化（独立）"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        raw_error = A - B
        best_error = best_preds - B
        
        ax.plot(np.abs(raw_error), alpha=0.5, color='gray', linewidth=1, label='校正前', zorder=1)
        ax.plot(np.abs(best_error), alpha=0.8, color='dodgerblue', linewidth=1.5, label=f'{best_model_name}校正后', zorder=2)
        ax.fill_between(range(len(best_error)), np.abs(best_error), alpha=0.08, color='dodgerblue', zorder=1)
        
        ax.set_xlabel('时间步', fontsize=12, fontweight='bold')
        ax.set_ylabel('|误差| (mm)', fontsize=12, fontweight='bold')
        ax.set_title('绝对误差时序变化', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_scatter_before(A, B, save_path="06_scatter_before.png"):
        """图6-1：校正前散点图（独立）"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sample_size = min(5000, len(A))
        sample_idx = np.linspace(0, len(A)-1, sample_size, dtype=int)
        
        ax.scatter(A[sample_idx], B[sample_idx], alpha=0.4, s=10, c=range(len(sample_idx)), cmap='viridis', zorder=2)
        ax.plot([A.min(), A.max()], [A.min(), A.max()], 'r--', alpha=0.6, linewidth=2.5, label='y=x (理想线)', zorder=3)
        
        corr = np.corrcoef(A[sample_idx], B[sample_idx])[0, 1]
        ax.set_title(f'校正前: A vs B  (相关系数 r={corr:.4f})', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('原始光纤数据A (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('基准数据B (mm)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_scatter_ridge(xgb_preds, B, save_path="06_scatter_ridge.png"):
        """图6-2：Ridge校正后散点图（独立）"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sample_size = min(5000, len(B))
        sample_idx = np.linspace(0, len(B)-1, sample_size, dtype=int)
        
        ax.scatter(xgb_preds[sample_idx], B[sample_idx], alpha=0.4, s=10, c=range(len(sample_idx)), cmap='viridis', zorder=2)
        ax.plot([B.min(), B.max()], [B.min(), B.max()], 'r--', alpha=0.6, linewidth=2.5, label='y=x (理想线)', zorder=3)
        
        corr = np.corrcoef(xgb_preds[sample_idx], B[sample_idx])[0, 1]
        ax.set_title(f'Ridge校正后: A\' vs B  (相关系数 r={corr:.4f})', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('校正后数据A\' (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('基准数据B (mm)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()

    @staticmethod
    def plot_scatter_best(best_preds, B, best_model_name="最佳模型", save_path="06_scatter_best.png"):
        """图6-3：最佳模型校正后散点图（独立）"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sample_size = min(5000, len(B))
        sample_idx = np.linspace(0, len(B)-1, sample_size, dtype=int)
        
        ax.scatter(best_preds[sample_idx], B[sample_idx], alpha=0.4, s=10, c=range(len(sample_idx)), cmap='viridis', zorder=2)
        ax.plot([B.min(), B.max()], [B.min(), B.max()], 'r--', alpha=0.6, linewidth=2.5, label='y=x (理想线)', zorder=3)
        
        corr = np.corrcoef(best_preds[sample_idx], B[sample_idx])[0, 1]
        ax.set_title(f'{best_model_name}校正后: A\' vs B  (相关系数 r={corr:.4f})', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('校正后数据A\' (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('基准数据B (mm)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()
