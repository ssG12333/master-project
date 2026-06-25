import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert, stft
from tqdm import tqdm
import json
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 25
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 24
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
SOURCE_DATA_FILE = 'final_source_data.parquet'
TARGET_DATA_FILE = 'final_target_data.parquet'
SELECTED_FEATURES_FILE = 'selected_features.json'
OUTPUT_PLOT_DIR = 'plots'
OUTPUT_SPECTROGRAM_DIR = 'spectrograms'
TARGET_FS = 32000
MAX_SPECTROGRAMS_PER_CLASS = 10
MAX_SPECTROGRAMS_TARGET = 50
PARAMS_DE = {'n_balls': 9, 'd_ball': 0.3126, 'd_pitch': 1.537}
PARAMS_FE = {'n_balls': 9, 'd_ball': 0.2656, 'd_pitch': 1.122}
def calculate_fault_frequencies(rpm, sensor):
    if rpm is None or pd.isna(rpm):
        return {'BPFO': 0, 'BPFI': 0, 'BSF': 0}
    params = PARAMS_DE if sensor == 'DE' else PARAMS_FE
    fr = rpm / 60.0
    nb = params['n_balls']
    d = params['d_ball']
    D = params['d_pitch']
    bpfo = (nb / 2.0) * fr * (1 - (d / D))
    bpfi = (nb / 2.0) * fr * (1 + (d / D))
    bsf = (D / (2.0 * d)) * fr * (1 - (d / D) ** 2)
    return {'BPFO': bpfo, 'BPFI': bpfi, 'BSF': bsf}
def plot_dataset_distribution(df):
    print("正在生成最终源域数据集类别分布图...")
    plt.figure(figsize=(14, 10))
    ax = sns.countplot(y=df['fault_type'], order=df['fault_type'].value_counts().index, palette='viridis')
    ax.set_title('最终筛选后源域数据集类别分布', fontsize=24)
    ax.set_xlabel('样本数量', fontsize=20)
    ax.set_ylabel('故障类别', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 50, p.get_y() + p.get_height() / 2, f'{int(width)}', ha='left', va='center', fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_PLOT_DIR, "final_dataset_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"数据集分布图已保存至: {save_path}")
def plot_four_class_comparison(df):
    print("正在生成四类典型信号综合对比图...")
    fault_types = ['N', 'B', 'IR', 'OR']
    if not all(ft in df['fault_type'].unique() for ft in fault_types):
        print("警告：数据集中未包含所有四种故障类型，无法生成对比图。")
        return
    fig, axes = plt.subplots(len(fault_types), 3, figsize=(24, 22))
    fig.suptitle('四类典型信号多维度对比分析', fontsize=28)
    for i, fault_type in enumerate(fault_types):
        sample_pool = df[(df['fault_type'] == fault_type) & (df['sensor'] != 'BA')]
        if sample_pool.empty:
            print(f"警告: 找不到类型为 {fault_type} 且非BA传感器的样本。")
            for j in range(3):
                axes[i, j].text(0.5, 0.5, f'无样本: {fault_type}', ha='center', va='center', fontsize=18)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            continue
        sample = sample_pool.iloc[0]
        segment = np.array(sample['signal_segment'])
        rpm = sample['rpm']
        sensor = sample['sensor']
        ax_time = axes[i, 0]
        time_axis = np.arange(len(segment)) / TARGET_FS
        ax_time.plot(time_axis, segment)
        ax_time.set_title(f'{fault_type} - 时域波形', fontsize=20)
        ax_time.grid(True)
        ax_time.tick_params(axis='both', which='major', labelsize=14)
        ax_env = axes[i, 1]
        envelope = np.abs(hilbert(segment - np.mean(segment)))
        N = len(envelope)
        fft_env = np.fft.fft(envelope)[:N // 2]
        fft_freqs = np.fft.fftfreq(N, 1 / TARGET_FS)[:N // 2]
        ax_env.plot(fft_freqs, np.abs(fft_env))
        ax_env.set_title(f'{fault_type} - 包络谱', fontsize=20)
        ax_env.set_xlim(0, 500)
        ax_env.grid(True)
        ax_env.tick_params(axis='both', which='major', labelsize=14)
        if fault_type != 'N':
            fault_freqs = calculate_fault_frequencies(rpm, sensor)
            colors = {'BPFO': 'r', 'BPFI': 'g', 'BSF': 'm'}
            key_map = {'OR': 'BPFO', 'IR': 'BPFI', 'B': 'BSF'}
            target_freq = fault_freqs[key_map[fault_type]]
            for j in range(1, 5):
                ax_env.axvline(x=target_freq * j, color=colors[key_map[fault_type]], linestyle='--',
                               label=f'{j}x{key_map[fault_type]}' if j == 1 else None)
            if any(v > 0 for v in fault_freqs.values()):
                ax_env.legend(fontsize=16)
        ax_spec = axes[i, 2]
        f, t, Zxx = stft(segment, fs=TARGET_FS, nperseg=256, noverlap=128)
        ax_spec.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
        ax_spec.set_title(f'{fault_type} - 时频谱图', fontsize=20)
        ax_spec.set_ylabel('频率 (Hz)', fontsize=18)
        ax_spec.set_xlabel('时间 (s)', fontsize=18)
        ax_spec.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = os.path.join(OUTPUT_PLOT_DIR, "four_class_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"四类对比图已保存至: {save_path}")
def generate_and_save_spectrograms(df, domain='source', max_samples_source=10, max_samples_target=50):
    print(f"正在为 {domain} 域生成并保存部分时频谱图...")
    base_dir = os.path.join(OUTPUT_SPECTROGRAM_DIR, domain)
    if domain == 'source':
        df_to_process = df.groupby('fault_type').head(max_samples_source)
        for fault_type in df['fault_type'].unique():
            os.makedirs(os.path.join(base_dir, fault_type), exist_ok=True)
    else:
        df_to_process = df.head(max_samples_target)
        os.makedirs(base_dir, exist_ok=True)
    print(f"将为 {domain} 域生成 {len(df_to_process)} 张时频谱图...")
    for index, row in tqdm(df_to_process.iterrows(), total=df_to_process.shape[0], desc=f"生成 {domain} 时频谱"):
        segment = np.array(row['signal_segment'])
        fig = plt.figure(frameon=False, figsize=(1, 1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        f, t, Zxx = stft(segment, fs=TARGET_FS, nperseg=256, noverlap=128)
        ax.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
        if domain == 'source':
            save_dir = os.path.join(base_dir, row['fault_type'])
        else:
            save_dir = base_dir
        original_index = row.name
        save_path = os.path.join(save_dir, f"{os.path.splitext(row['filename'])[0]}_sample_{original_index}.png")
        plt.savefig(save_path, dpi=128)
        plt.close(fig)
def plot_source_target_feature_comparison(source_df, target_df, selected_features):
    print("正在生成源域与目标域的特征分布对比图...")
    feature_comp_dir = os.path.join(OUTPUT_PLOT_DIR, 'source_target_comparison')
    os.makedirs(feature_comp_dir, exist_ok=True)
    print("  - 生成雷达图...")
    stats_source = source_df[selected_features].mean()
    stats_target = target_df[selected_features].mean()
    scaler = MinMaxScaler()
    stats_scaled = scaler.fit_transform(pd.concat([stats_source, stats_target], axis=1))
    labels = selected_features
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(polar=True))
    values_source = np.concatenate((stats_scaled[:, 0], stats_scaled[:1, 0]))
    ax.plot(angles, values_source, color='blue', linewidth=1.5, linestyle='solid', label='源域 (筛选后)')
    ax.fill(angles, values_source, color='blue', alpha=0.25)
    values_target = np.concatenate((stats_scaled[:, 1], stats_scaled[:1, 1]))
    ax.plot(angles, values_target, color='orangered', linewidth=1.5, linestyle='solid', label='目标域')
    ax.fill(angles, values_target, color='orangered', alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=18)
    ax.set_title('源域与目标域核心特征宏观对比', size=26, color='black', y=1.1)
    save_path = os.path.join(feature_comp_dir, "st_comparison_radar.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  雷达图已保存至: {save_path}")
    # --- 2. 分面箱形图 ---
    print("  - 生成分面箱形图...")
    source_plot_df = source_df[selected_features].copy()
    source_plot_df['domain'] = '源域 (筛选后)'
    target_plot_df = target_df[selected_features].copy()
    target_plot_df['domain'] = '目标域'
    combined_df = pd.concat([source_plot_df, target_plot_df])
    melted_df = combined_df.melt(id_vars=['domain'], var_name='特征', value_name='特征值')
    g = sns.catplot(x='domain', y='特征值', col='特征', data=melted_df, kind='box',
                    col_wrap=5, height=5, aspect=1.3, sharey=False, palette='muted')
    g.fig.suptitle('源域与目标域核心特征微观对比', y=1.03, fontsize=26)
    g.set_titles("{col_name}", size=18)
    g.set_axis_labels("域", "特征值", fontsize=20)
    g.tick_params(axis='both', which='major', labelsize=14)
    save_path = os.path.join(feature_comp_dir, "st_comparison_boxplots.png")
    g.savefig(save_path)
    plt.close()
    print(f"  分面箱形图已保存至: {save_path}")
if __name__ == '__main__':
    if not os.path.exists(SOURCE_DATA_FILE) or not os.path.exists(SELECTED_FEATURES_FILE):
        print("错误：无法找到`final_source_data.parquet`或`selected_features.json`。")
        print("请确保您已成功运行更新版的`第一问第二阶段.py`脚本。")
    else:
        print(f"正在加载 {SOURCE_DATA_FILE}...")
        source_df = pd.read_parquet(SOURCE_DATA_FILE)

        print(f"正在加载 {TARGET_DATA_FILE}...")
        target_df = pd.read_parquet(TARGET_DATA_FILE)

        print(f"正在加载 {SELECTED_FEATURES_FILE}...")
        with open(SELECTED_FEATURES_FILE, 'r') as f:
            selected_features = json.load(f)

        if not os.path.exists(OUTPUT_PLOT_DIR):
            os.makedirs(OUTPUT_PLOT_DIR)
        if not os.path.exists(OUTPUT_SPECTROGRAM_DIR):
            os.makedirs(OUTPUT_SPECTROGRAM_DIR)
        plot_dataset_distribution(source_df)
        plot_four_class_comparison(source_df)
        plot_source_target_feature_comparison(source_df, target_df, selected_features)
        generate_and_save_spectrograms(source_df, domain='source', max_samples_source=MAX_SPECTROGRAMS_PER_CLASS)
        generate_and_save_spectrograms(target_df, domain='target', max_samples_target=MAX_SPECTROGRAMS_TARGET)
        print("\n所有最终可视化任务已完成！")

