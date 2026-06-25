import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from datetime import datetime, timedelta

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

class Q2Visualizer:
    @staticmethod
    def _get_datetime_axis(df):
        start_date = datetime(2024, 5, 4, 0, 0, 0)
        times = [start_date + timedelta(minutes=10*i) for i in range(len(df))]
        return times

    @staticmethod
    def plot_raw_data(df, save_path="q2_fig1_raw_data.png"):
        """图1: 附件2原始位移时序数据展示"""
        fig, ax = plt.subplots(figsize=(14, 5))
        time_hours = np.arange(len(df)) / 6.0
        ax.plot(time_hours, df['Displacement'], color='steelblue', linewidth=0.8)
        ax.set_xlabel('时间 (小时, 从2024-05-04 00:00起)', fontsize=12)
        ax.set_ylabel('表面位移 (mm)', fontsize=12)
        ax.set_title('图1: 附件2 原始表面位移时序数据', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        start_str = '2024-05-04 00:00'
        end_hours = time_hours[-1]
        end_days = int(end_hours // 24)
        end_h = int(end_hours % 24)
        end_str = f'2024-05-{4+end_days} {end_h:02d}:00'
        ax.text(0.02, 0.95, f'起止时间: {start_str} ~ {end_str}', 
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.02, 0.88, f'采集频率: 10分钟/次', 
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"-> 成功生成: {save_path}")
        plt.close()

    @staticmethod
    def plot_all_stages_combined(df, model_results, save_path="q2_fig12_all_stages_combined.png"):
        """图12: 三阶段数学模型整体效果展示 (统一横轴)"""
        fig, ax = plt.subplots(figsize=(16, 8))
        dt = 1.0 / 6.0
        cp1, cp2 = model_results['cp1'], model_results['cp2']
        t_all = np.arange(len(df)) * dt
        x_all = df['Disp_Smooth'].values
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        model_colors = ['#1a8a4a', '#d4860a', '#c0392b']
        labels = ['阶段1 实测', '阶段2 实测', '阶段3 实测']
        model_labels = ['阶段1 线性拟合', '阶段2 幂函数拟合', '阶段3 Voight拟合']
        
        ax.plot(t_all, x_all, 'b-', linewidth=1.5, alpha=0.6, label='全时段平滑位移')
        
        t1 = np.arange(cp1) * dt
        x1 = x_all[:cp1]
        ax.scatter(t1, x1, s=1, color=colors[0], alpha=0.5, label=labels[0])
        if model_results['stage1']['params']:
            k, c = model_results['stage1']['params']['k'], model_results['stage1']['params']['c']
            ax.plot(t1, k*t1+c, color=model_colors[0], linewidth=2.5, linestyle='--', label=model_labels[0])
        
        t2 = np.arange(cp1, cp2) * dt
        x2 = x_all[cp1:cp2]
        t2_local = t2 - t2[0]
        ax.scatter(t2, x2, s=1, color=colors[1], alpha=0.5, label=labels[1])
        if model_results['stage2']['params']:
            A, B, C = model_results['stage2']['params']['A'], model_results['stage2']['params']['B'], model_results['stage2']['params']['C']
            ax.plot(t2, A*np.maximum(t2_local, 0.1)**B + C, color=model_colors[1], linewidth=2.5, linestyle='--', label=model_labels[1])
        
        t3 = np.arange(cp2, len(df)) * dt
        x3 = x_all[cp2:]
        t3_local = t3 - t3[0]
        ax.scatter(t3, x3, s=1, color=colors[2], alpha=0.5, label=labels[2])
        if model_results['stage3']['params']:
            from q2_models import voight_model
            try:
                tf, lam, C = model_results['stage3']['params']['tf'], model_results['stage3']['params']['lambda'], model_results['stage3']['params']['C']
                ax.plot(t3, voight_model(t3_local, tf, lam, C) + x3[0] - voight_model(0, tf, lam, C), 
                       color=model_colors[2], linewidth=2.5, linestyle='--', label=model_labels[2])
            except:
                pass
        
        ax.axvline(cp1*dt, color='k', linestyle='-.', lw=2, alpha=0.6, label=f'tc1={cp1*dt:.0f}h')
        ax.axvline(cp2*dt, color='k', linestyle='-.', lw=2, alpha=0.6, label=f'tc2={cp2*dt:.0f}h')
        
        # 信息框放在右上角
        info_text = (f'tc1 = {cp1} ({cp1*dt:.1f}h)  |  tc2 = {cp2} ({cp2*dt:.1f}h)\n'
                     f'R$^2$_1={model_results["stage1"]["r2"]:.3f}  R$^2$_2={model_results["stage2"]["r2"]:.4f}  R$^2$_3={model_results["stage3"]["r2"]:.3f}')
        ax.text(0.98, 0.97, info_text, transform=ax.transAxes, fontsize=11, 
                va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel("时间 (小时, 从观测起点)", fontsize=13)
        ax.set_ylabel("表面位移 (mm)", fontsize=13)
        ax.set_title('图12: 三阶段数学模型整体拟合效果 (统一时间轴)', fontsize=14)
        # 图例放在图表上方空白区域
        ax.legend(loc='upper center', fontsize=8, ncol=4, framealpha=0.8, bbox_to_anchor=(0.5, 1.15))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(save_path, dpi=300)
        print(f"-> 成功生成: {save_path}")
        plt.close()

    @staticmethod
    def plot_comprehensive_judgment(df, cp1, cp2, save_path="q2_fig11_comprehensive.png"):
        """图11: 多尺度阶段切分综合判别图 (位移、速度、切线角三图联动)"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        time_hours = np.arange(len(df)) / 6.0
        tc1, tc2 = time_hours[cp1], time_hours[cp2]
        
        spans = [(0, tc1, '#eaffea', '阶段1'), 
                 (tc1, tc2, '#fff4e6', '阶段2'), 
                 (tc2, time_hours[-1], '#ffebeb', '阶段3')]
        
        axes[0].plot(time_hours, df['Displacement'], color='gray', alpha=0.5, lw=0.8, label='原始数据 (含跳变)')
        axes[0].plot(time_hours, df['Disp_Smooth'], color='blue', linewidth=2, label='平滑真实形变')
        axes[0].set_ylabel("表面位移 (mm)", fontsize=12)
        axes[0].set_title("图11: 多尺度阶段切分综合判别图\n(a) 位移演化全貌与阶段划分", fontsize=14)
        axes[0].legend(loc='upper left', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(time_hours, df['Velocity'], color='orange', linewidth=1.0, label='瞬时速度')
        axes[1].set_ylabel("速度 (mm/h)", fontsize=12)
        axes[1].set_title("(b) 瞬时速度演化特征", fontsize=13)
        axes[1].grid(True, alpha=0.3)
        
        v0 = np.mean(df['Velocity'].values[:200])
        vel_smooth = gaussian_filter1d(df['Velocity'].values, sigma=50)
        vel_smooth = np.maximum(vel_smooth, 1e-8)
        tangent_angle = np.degrees(np.arctan(vel_smooth / v0))
        
        axes[2].plot(time_hours, tangent_angle, color='purple', linewidth=1.0, label='宏观切线角')
        axes[2].axhline(45, color='black', linestyle='--', alpha=0.6, label='45° 阈值线')
        axes[2].axhline(80, color='red', linestyle='--', alpha=0.6, label='80° 阈值线')
        axes[2].set_ylabel("切线角 (°)", fontsize=12)
        axes[2].set_xlabel("时间 (小时)", fontsize=12)
        axes[2].set_title("(c) 切线角阈值演变监测", fontsize=13)
        axes[2].legend(loc='upper left', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        for ax in axes:
            for start, end, color, name in spans:
                ax.axvspan(start, end, facecolor=color, alpha=0.4)
            ax.axvline(tc1, color='red', linestyle='-.', lw=2)
            ax.axvline(tc2, color='red', linestyle='-.', lw=2)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"-> 成功生成: {save_path}")
        plt.close()

    @staticmethod
    def plot_preprocessing_comparison(df, save_path="q2_fig2_preprocessing.png"):
        """图2: 数据预处理三步效果对比 (原始→清洗→平滑)"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        time_hours = np.arange(len(df)) / 6.0
        
        axes[0].plot(time_hours, df['Displacement'], color='steelblue', linewidth=0.8)
        axes[0].set_ylabel("位移 (mm)", fontsize=11)
        axes[0].set_title("(a) 原始位移数据 (含瞬时跳变噪声)", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(time_hours, df['Disp_Clean'], color='darkgreen', linewidth=1.0)
        axes[1].set_ylabel("位移 (mm)", fontsize=11)
        axes[1].set_title("(b) Hampel滤波剔除脉冲噪声后", fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(time_hours, df['Disp_Smooth'], color='crimson', linewidth=1.5)
        axes[2].set_ylabel("位移 (mm)", fontsize=11)
        axes[2].set_xlabel("时间 (小时)", fontsize=12)
        axes[2].set_title("(c) SG滤波平滑+单调递增约束后", fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"-> 成功生成: {save_path}")
        plt.close()

    @staticmethod
    def plot_noise_vs_real_transition(df, save_path="q2_fig3_noise_discrimination.png"):
        """图3: 噪声跳变 vs 真实阶段转换 核心判别准则"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        time_hours = np.arange(len(df)) / 6.0
        
        vel = df['Velocity'].values
        vel_smooth = gaussian_filter1d(vel, sigma=50)
        vel_smooth = np.maximum(vel_smooth, 1e-8)
        
        axes[0, 0].plot(time_hours[:200], vel[:200], 'gray', alpha=0.5, lw=0.5)
        axes[0, 0].set_title('(a) 噪声区域: 速度瞬时跳变后迅速恢复', fontsize=11)
        axes[0, 0].set_ylabel("速度 (mm/h)", fontsize=10)
        axes[0, 0].set_xlabel("时间 (小时)", fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(time_hours[:200], vel_smooth[:200], 'r-', lw=1.5)
        axes[0, 1].axhline(0, color='k', ls='-', lw=0.3)
        axes[0, 1].set_title('(b) 噪声特征: 速度波动无持续趋势', fontsize=11)
        axes[0, 1].set_ylabel("平滑速度 (mm/h)", fontsize=10)
        axes[0, 1].set_xlabel("时间 (小时)", fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        cp1 = 8043
        window = 1000
        axes[1, 0].plot(time_hours[cp1-window:cp1+window], vel_smooth[cp1-window:cp1+window], 'b-', lw=1.5)
        axes[1, 0].axvline(time_hours[cp1], color='red', ls='--', lw=2, label='tc1')
        axes[1, 0].set_title(f'(c) 真实转换: 速度持续跃升 (tc1={cp1})', fontsize=11)
        axes[1, 0].set_ylabel("速度 (mm/h)", fontsize=10)
        axes[1, 0].set_xlabel("时间 (小时)", fontsize=10)
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        
        log_vel = np.log(vel_smooth)
        log_vel_s = gaussian_filter1d(log_vel, sigma=100)
        log_vel_d = np.gradient(log_vel_s)
        axes[1, 1].plot(time_hours[cp1-window:cp1+window], log_vel_d[cp1-window:cp1+window], 'purple', lw=1.5)
        axes[1, 1].axvline(time_hours[cp1], color='red', ls='--', lw=2, label='tc1')
        axes[1, 1].axhline(0, color='k', ls='-', lw=0.3)
        axes[1, 1].set_title('(d) 真实转换: 对数速度斜率持续增大', fontsize=11)
        axes[1, 1].set_ylabel("d[ln(v)]/dt", fontsize=10)
        axes[1, 1].set_xlabel("时间 (小时)", fontsize=10)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"-> 成功生成: {save_path}")
        plt.close()

    @staticmethod
    def plot_tc1_identification(df, cp1, save_path="q2_fig4_tc1.png"):
        """图4: 转换节点tc1识别 (缓慢匀速→加速形变)"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        time_hours = np.arange(len(df)) / 6.0
        
        vel = df['Velocity'].values
        vel_smooth = gaussian_filter1d(vel, sigma=30)
        vel_smooth = np.maximum(vel_smooth, 1e-8)
        
        window = 2000
        start = max(0, cp1 - window)
        end = min(len(df), cp1 + window)
        
        axes[0].plot(time_hours[start:end], vel_smooth[start:end], 'darkblue', lw=2)
        axes[0].axvline(time_hours[cp1], color='red', ls='--', lw=2, label=f'tc1 = {cp1} ({cp1/6:.1f}h)')
        axes[0].set_ylabel("速度 (mm/h)", fontsize=11)
        axes[0].set_title('图4: 转换节点 tc1 识别 (缓慢匀速 → 加速形变)', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        v0 = np.mean(vel_smooth[:200])
        vel_ratio = vel_smooth / v0
        axes[1].plot(time_hours[start:end], vel_ratio[start:end], 'crimson', lw=2)
        axes[1].axvline(time_hours[cp1], color='red', ls='--', lw=2, label=f'tc1 = {cp1} ({cp1/6:.1f}h)')
        axes[1].axhline(1, color='k', ls='--', lw=1, alpha=0.3)
        axes[1].set_ylabel("速度比率 v/v₀", fontsize=11)
        axes[1].set_xlabel("时间 (小时)", fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"-> 成功生成: {save_path}")
        plt.close()

    @staticmethod
    def plot_tc2_identification(df, cp1, cp2, save_path="q2_fig5_tc2.png"):
        """图5: 转换节点tc2识别 (加速形变→快速失稳)"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        time_hours = np.arange(len(df)) / 6.0
        
        vel = df['Velocity'].values
        vel_smooth = gaussian_filter1d(vel, sigma=100)
        vel_smooth = np.maximum(vel_smooth, 1e-8)
        
        window = 1500
        start = max(0, cp2 - window)
        end = min(len(df), cp2 + 200)
        
        axes[0].plot(time_hours[start:end], vel_smooth[start:end], 'darkorange', lw=2)
        axes[0].axvline(time_hours[cp2], color='red', ls='--', lw=2, label=f'tc2 = {cp2} ({cp2/6:.1f}h)')
        axes[0].set_ylabel("速度 (mm/h)", fontsize=11)
        axes[0].set_title('图5: 转换节点 tc2 识别 (加速形变 → 快速失稳)', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        vel_dd = np.gradient(np.gradient(vel_smooth))
        vel_dd_smooth = gaussian_filter1d(vel_dd, sigma=200)
        axes[1].plot(time_hours[start:end], vel_dd_smooth[start:end], 'purple', lw=1.5)
        axes[1].axvline(time_hours[cp2], color='red', ls='--', lw=2, label=f'tc2 = {cp2} ({cp2/6:.1f}h)')
        axes[1].axhline(0, color='k', ls='-', lw=0.5)
        axes[1].set_ylabel("速度二阶导 d²v/dt²", fontsize=11)
        axes[1].set_xlabel("时间 (小时)", fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"-> 成功生成: {save_path}")
        plt.close()

    @staticmethod
    def plot_stage1_model(df, model_results, save_path="q2_fig6_stage1.png"):
        """图6: 阶段1 (缓慢匀速期) 数学模型"""
        fig, ax = plt.subplots(figsize=(10, 6))
        dt = 1.0 / 6.0
        cp1 = model_results['cp1']
        t1 = np.arange(cp1) * dt
        x1 = df['Disp_Smooth'].values[:cp1]
        s1 = model_results['stage1']
        
        ax.plot(t1, x1, 'b.', markersize=2, alpha=0.5, label='实测数据')
        if s1['params']:
            k, c = s1['params']['k'], s1['params']['c']
            ax.plot(t1, k*t1+c, 'r-', linewidth=2, label=f"拟合: x(t)={k:.4f}t+{c:.2f}")
        
        duration = cp1 * dt
        dx = df['Disp_Smooth'].iloc[cp1-1] - df['Disp_Smooth'].iloc[0]
        avg_v = dx / duration if duration > 0 else 0
        
        info_text = (f'R$^2$ = {s1["r2"]:.4f}\n'
                     f'RMSE = {s1["rmse"]:.2f} mm\n'
                     f'时长 = {duration:.1f} h\n'
                     f'位移变化 = {dx:.2f} mm\n'
                     f'平均速度 = {avg_v:.4f} mm/h')
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
                va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel("时间 (小时)", fontsize=12)
        ax.set_ylabel("表面位移 (mm)", fontsize=12)
        ax.set_title('图6: 阶段1 (缓慢匀速期) — 线性模型', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"-> 成功生成: {save_path}")
        plt.close()

    @staticmethod
    def plot_stage2_model(df, model_results, save_path="q2_fig7_stage2.png"):
        """图7: 阶段2 (加速形变期) 数学模型"""
        fig, ax = plt.subplots(figsize=(10, 6))
        dt = 1.0 / 6.0
        cp1, cp2 = model_results['cp1'], model_results['cp2']
        t2 = np.arange(cp1, cp2) * dt
        x2 = df['Disp_Smooth'].values[cp1:cp2]
        s2 = model_results['stage2']
        t2_local = t2 - t2[0]
        
        ax.plot(t2_local, x2, 'b.', markersize=2, alpha=0.5, label='实测数据')
        if s2['params']:
            A, B, C = s2['params']['A'], s2['params']['B'], s2['params']['C']
            ax.plot(t2_local, A*np.maximum(t2_local, 0.1)**B + C, 'r-', linewidth=2, 
                    label=f"拟合: x(t)={A:.4f}t^{B:.2f}+{C:.1f}")
        
        duration = (cp2 - cp1) * dt
        dx = df['Disp_Smooth'].iloc[cp2-1] - df['Disp_Smooth'].iloc[cp1]
        avg_v = dx / duration if duration > 0 else 0
        
        info_text = (f'R$^2$ = {s2["r2"]:.4f}\n'
                     f'RMSE = {s2["rmse"]:.2f} mm\n'
                     f'时长 = {duration:.1f} h\n'
                     f'位移变化 = {dx:.2f} mm\n'
                     f'平均速度 = {avg_v:.4f} mm/h')
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
                va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel("时间 (小时, 从阶段起点)", fontsize=12)
        ax.set_ylabel("表面位移 (mm)", fontsize=12)
        ax.set_title('图7: 阶段2 (加速形变期) — 幂函数模型', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"-> 成功生成: {save_path}")
        plt.close()

    @staticmethod
    def plot_stage3_model(df, model_results, save_path="q2_fig8_stage3.png"):
        """图8: 阶段3 (快速失稳期) 数学模型"""
        fig, ax = plt.subplots(figsize=(10, 6))
        dt = 1.0 / 6.0
        cp2 = model_results['cp2']
        t3 = np.arange(cp2, len(df)) * dt
        x3 = df['Disp_Smooth'].values[cp2:]
        s3 = model_results['stage3']
        t3_local = t3 - t3[0]
        
        ax.plot(t3_local, x3, 'b.', markersize=2, alpha=0.5, label='实测数据')
        if s3['params']:
            from q2_models import voight_model
            try:
                tf, lam, C = s3['params']['tf'], s3['params']['lambda'], s3['params']['C']
                ax.plot(t3_local, voight_model(t3_local, tf, lam, C) + x3[0] - voight_model(0, tf, lam, C), 
                       'r-', linewidth=2, label=f"Voight拟合 (tf={tf:.1f}h)")
            except:
                pass
        
        duration = (len(df) - cp2) * dt
        dx = df['Disp_Smooth'].iloc[-1] - df['Disp_Smooth'].iloc[cp2]
        avg_v = dx / duration if duration > 0 else 0
        
        tf_abs = cp2 * dt + s3['params'].get('tf', 0)
        info_text = (f'R$^2$ = {s3["r2"]:.4f}\n'
                     f'RMSE = {s3["rmse"]:.2f} mm\n'
                     f'时长 = {duration:.1f} h\n'
                     f'位移变化 = {dx:.2f} mm\n'
                     f'平均速度 = {avg_v:.4f} mm/h\n'
                     f'预测失稳: {tf_abs:.1f}h')
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
                va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel("时间 (小时, 从阶段起点)", fontsize=12)
        ax.set_ylabel("表面位移 (mm)", fontsize=12)
        ax.set_title('图8: 阶段3 (快速失稳期) — Voight/Saito模型', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"-> 成功生成: {save_path}")
        plt.close()

    @staticmethod
    def plot_stage_colored_displacement(df, cp1, cp2, save_path="q2_fig9_stage_division.png"):
        """图9: 三阶段位移时序自适应划分"""
        fig, ax = plt.subplots(figsize=(14, 5))
        time_hours = np.arange(len(df)) / 6.0
        disp = df['Disp_Smooth'].values
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        segments = [(0, cp1), (cp1, cp2), (cp2, len(df))]
        
        for (start, end), color in zip(segments, colors):
            ax.scatter(time_hours[start:end], disp[start:end], s=1.5, color=color, alpha=0.8)
        
        ax.set_xlabel('时间 (小时)', fontsize=12)
        ax.set_ylabel('表面位移 (mm)', fontsize=12)
        ax.set_title('图9: 基于转换节点的三阶段位移自适应划分', fontsize=13)
        
        mean_v1 = np.mean(df['Velocity'].iloc[:cp1])
        mean_v2 = np.mean(df['Velocity'].iloc[cp1:cp2])
        mean_v3 = np.mean(df['Velocity'].iloc[cp2:])
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=6, 
                       label=f'阶段1 缓慢匀速 (v̄={mean_v1:.3f} mm/h)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=6,
                       label=f'阶段2 加速形变 (v̄={mean_v2:.3f} mm/h)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], markersize=6,
                       label=f'阶段3 快速失稳 (v̄={mean_v3:.3f} mm/h)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"-> 成功生成: {save_path}")
        plt.close()

    @staticmethod
    def plot_kde_comparison(df, cp1, cp2, save_path="q2_fig10_kde.png"):
        """图10: 不同演化阶段下表面位移速率的核密度分布对比"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        vel_smooth = gaussian_filter1d(df['Velocity'].values, sigma=50)
        v1 = vel_smooth[:cp1]
        v2 = vel_smooth[cp1:cp2]
        v3 = vel_smooth[cp2:]
        
        all_vel = np.concatenate([v1, v2, v3])
        x_grid = np.linspace(np.min(all_vel), np.max(all_vel), 200)
        
        from scipy.stats import gaussian_kde
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        labels = ['阶段1 缓慢匀速', '阶段2 加速形变', '阶段3 快速失稳']
        
        for v, color, label in zip([v1, v2, v3], colors, labels):
            kde = gaussian_kde(v)
            y = kde(x_grid)
            ax.fill_between(x_grid, y, alpha=0.3, color=color, label=label)
            ax.plot(x_grid, y, color=color, linewidth=2)
        
        ax.set_xlabel('形变速率 (mm/h)', fontsize=12)
        ax.set_ylabel('频数密度', fontsize=12)
        ax.set_title('图10: 不同演化阶段下表面位移速率的核密度分布对比', fontsize=13)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"-> 成功生成: {save_path}")
        plt.close()
