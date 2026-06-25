import os
import re
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pickle
from scipy.fft import fft
import pywt
from scipy.signal import resample

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 25
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['figure.titlesize'] = 24
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
DATA_ROOT_PATH = '数据集'
TARGET_FILENAME_PREFIXES = [chr(ord('A') + i) for i in range(16)]
TARGET_FS = 32000
PLOT_OUTPUT_DIR = 'plots'
OUTPUT_DATA_FILE = 'preprocessed_data.pkl'
DENOISE_THRESHOLD_SCALER = 0.5
SEGMENT_LENGTH = 4096
OVERLAP_RATIO = 0.5
def plot_comparison(original_signal, denoised_signal, final_processed_signal, original_fs, target_fs, filename, sensor):
    print(f"  正在为 {filename} ({sensor}) 生成三部分对比图...")
    if not os.path.exists(PLOT_OUTPUT_DIR):
        os.makedirs(PLOT_OUTPUT_DIR)
    fig, axs = plt.subplots(3, 1, figsize=(15, 18))
    plot_len_orig = min(len(original_signal), 4096)
    time_axis_original = np.arange(plot_len_orig) / original_fs
    axs[0].plot(time_axis_original, original_signal[:plot_len_orig], label='原始信号 (Original)', alpha=0.7)
    axs[0].plot(time_axis_original, denoised_signal[:plot_len_orig], label='去噪后信号 (Denoised)', alpha=0.9)
    axs[0].set_title(f'时域对比 (前 {plot_len_orig / original_fs:.4f} 秒) - {filename} [{sensor}]', fontweight='bold')
    axs[0].set_xlabel('时间 (s)', fontweight='bold')
    axs[0].set_ylabel('振幅', fontweight='bold')
    axs[0].legend()
    axs[0].grid(True)
    plot_len_final = min(len(final_processed_signal), 4096)
    time_axis_final = np.arange(plot_len_final) / target_fs
    axs[1].plot(time_axis_final, final_processed_signal[:plot_len_final], label='去噪+重采样后信号', color='orangered')
    axs[1].set_title(f'最终处理后时域信号 (前 {plot_len_final / target_fs:.4f} 秒) - {filename} [{sensor}]', fontweight='bold')
    axs[1].set_xlabel('时间 (s)', fontweight='bold')
    axs[1].set_ylabel('振幅', fontweight='bold')
    axs[1].legend()
    axs[1].grid(True)
    def to_db(amplitude):
        return 20 * np.log10(np.where(amplitude <= 0, 1e-12, amplitude))
    N_orig = len(original_signal)
    yf_orig = fft(original_signal)
    amp_orig = 2.0 / N_orig * np.abs(yf_orig[0:N_orig // 2])
    db_orig = to_db(amp_orig)
    xf_orig = np.linspace(0.0, original_fs / 2.0, N_orig // 2)

    N_denoised = len(denoised_signal)
    yf_denoised = fft(denoised_signal)
    amp_denoised = 2.0 / N_denoised * np.abs(yf_denoised[0:N_denoised // 2])
    db_denoised = to_db(amp_denoised)
    xf_denoised = np.linspace(0.0, original_fs / 2.0, N_denoised // 2)
    N_final = len(final_processed_signal)
    yf_final = fft(final_processed_signal)
    amp_final = 2.0 / N_final * np.abs(yf_final[0:N_final // 2])
    db_final = to_db(amp_final)
    xf_final = np.linspace(0.0, target_fs / 2.0, N_final // 2)
    axs[2].plot(xf_orig, db_orig, label=f'原始信号频谱 (Fs={original_fs}Hz)', alpha=0.5, color='blue')
    axs[2].plot(xf_denoised, db_denoised, label=f'去噪后频谱 (Fs={original_fs}Hz)', alpha=0.7, color='green')
    axs[2].plot(xf_final, db_final, label=f'去噪+重采样后频谱 (Fs={target_fs}Hz)', alpha=0.9, color='orangered')
    axs[2].set_title(f'频域对比 - {filename} [{sensor}]', fontweight='bold')
    axs[2].set_xlabel('频率 (Hz)', fontweight='bold')
    axs[2].set_ylabel('功率 (dB)', fontweight='bold')
    axs[2].set_xlim(0, target_fs / 2)
    axs[2].legend()
    axs[2].grid(True)
    plt.tight_layout()
    save_path = os.path.join(PLOT_OUTPUT_DIR, f"comparison_{os.path.splitext(filename)[0]}_{sensor}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  对比图已保存至: {save_path}")
def denoise_signal(signal, wavelet='db4', level=4, threshold_scaler=DENOISE_THRESHOLD_SCALER):
    coeff = pywt.wavedec(signal, wavelet, mode="per")
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal))) * threshold_scaler
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    denoised_signal = pywt.waverec(coeff, wavelet, mode='per')
    return denoised_signal[:len(signal)]
def resample_signal(signal, original_fs, target_fs):
    if original_fs == target_fs:
        return signal
    num_samples = int(len(signal) * float(target_fs) / original_fs)
    return resample(signal, num_samples)

def load_and_preprocess_data(root_path):
    source_data = []
    target_data = []
    plotted_source_counts = {'N': 0, 'IR': 0, 'OR': 0, 'B': 0}
    max_plot_per_type = 2
    print(f"开始从 '{root_path}' 文件夹加载数据...")
    for dirpath, _, filenames in os.walk(root_path):
        for filename in sorted(filenames):
            if filename.endswith('.mat'):
                file_path = os.path.join(dirpath, filename)
                try:
                    mat_data = scipy.io.loadmat(file_path)
                    print(f"\n--- 正在加载文件: {filename} ---")
                    print(f"  文件内变量 (Keys): {list(mat_data.keys())}")
                    filename_no_ext = os.path.splitext(filename)[0]
                    is_target = filename_no_ext in TARGET_FILENAME_PREFIXES
                    rpm = None
                    raw_signals = {}
                    if is_target:
                        if filename_no_ext in mat_data:
                            rpm = 600
                            raw_signals['target'] = mat_data[filename_no_ext].flatten()
                    else:
                        # --- RPM 智能提取逻辑 ---
                        # 1. 优先从mat文件内部读取
                        for key in mat_data.keys():
                            if 'RPM' in key:
                                rpm = mat_data[key][0][0]
                                break  # 找到即停止
                        if rpm is None:
                            match = re.search(r'\((\d+)rpm\)', filename)
                            if match:
                                rpm = int(match.group(1))
                                print(f"  已从文件名中成功解析RPM: {rpm}")
                        # 读取信号数据
                        for key in mat_data.keys():
                            if '_time' in key.lower() and not key.startswith('__'):
                                sensor_type = key.split('_')[-2].upper()
                                if sensor_type in ['DE', 'FE', 'BA']:
                                    raw_signals[sensor_type] = mat_data[key].flatten()
                    if not raw_signals:
                        print(f"  警告: 在文件 {filename} 中未找到可识别的信号数据。")
                        continue
                    processed_signals = {}
                    raw_signals_info = {}
                    fault_type = "UNKNOWN"
                    if not is_target:
                        fault_type_match = re.match(r"([a-zA-Z]+)", filename)
                        fault_type = fault_type_match.group(1).upper() if fault_type_match else "UNKNOWN"
                        if fault_type == 'NORMAL': fault_type = 'N'
                    for sensor, signal_array in raw_signals.items():
                        if is_target:
                            original_fs = 32000
                        else:
                            original_fs = 48000 if "48k" in dirpath.lower() else 12000
                        original_len = len(signal_array)
                        raw_signals_info[sensor] = {'length': original_len, 'fs': original_fs}
                        denoised = denoise_signal(signal_array)
                        resampled = resample_signal(denoised, original_fs=original_fs, target_fs=TARGET_FS)
                        processed_signals[sensor] = resampled
                        print(f"  传感器 '{sensor}': 原始长度={original_len} (Fs={original_fs}Hz) -> "
                              f"处理后长度={len(resampled)} (Fs={TARGET_FS}Hz)")
                        should_plot = False
                        if is_target:
                            should_plot = True
                        elif plotted_source_counts.get(fault_type, 0) < max_plot_per_type:
                            should_plot = True
                        if should_plot:
                            plot_comparison(signal_array, denoised, resampled, original_fs, TARGET_FS, filename, sensor)
                            if not is_target:
                                plotted_source_counts[fault_type] += 1
                    data_entry = {
                        'filename': filename, 'filepath': file_path, 'rpm': rpm,
                        'signals': processed_signals,
                        'raw_signals_info': raw_signals_info
                    }
                    if is_target:
                        target_data.append(data_entry)
                    else:
                        data_entry['fault_type'] = fault_type
                        source_data.append(data_entry)
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
    print(f"\n加载完成！源域文件数: {len(source_data)}, 目标域文件数: {len(target_data)}")
    return source_data, target_data

def segment_and_unify_data(dataset, is_source=True):
    segmented_data = []
    step = int(SEGMENT_LENGTH * (1 - OVERLAP_RATIO))
    for data_entry in dataset:
        for sensor, signal in data_entry['signals'].items():
            for i in range(0, len(signal) - SEGMENT_LENGTH + 1, step):
                segment = signal[i: i + SEGMENT_LENGTH]
                new_entry = {
                    'filename': data_entry['filename'],
                    'rpm': data_entry['rpm'],
                    'sensor': sensor,
                    'signal_segment': segment
                }
                if is_source:
                    new_entry['fault_type'] = data_entry['fault_type']
                segmented_data.append(new_entry)
    return segmented_data
if __name__ == '__main__':
    source_dataset, target_dataset = load_and_preprocess_data(DATA_ROOT_PATH)
    print("\n--- 开始对数据进行分段 ---")
    segmented_source_data = segment_and_unify_data(source_dataset, is_source=True)
    segmented_target_data = segment_and_unify_data(target_dataset, is_source=False)
    print("\n--- 数据分段完成 ---")
    print(f"源域文件 {len(source_dataset)} 个, 切分为 {len(segmented_source_data)} 个样本")
    print(f"目标域文件 {len(target_dataset)} 个, 切分为 {len(segmented_target_data)} 个样本")
    print(f"\n--- 正在将处理结果保存到 {OUTPUT_DATA_FILE} ---")
    with open(OUTPUT_DATA_FILE, 'wb') as f:
        pickle.dump((segmented_source_data, segmented_target_data), f)
    print("数据保存成功！")
    if segmented_source_data:
        print("\n--- 源域分段后首个样本示例 ---")
        first_sample = segmented_source_data[0]
        print(f"源文件: {first_sample['filename']}")
        print(f"故障类型: {first_sample['fault_type']}")
        print(f"传感器: {first_sample['sensor']}")
        print(f"样本长度: {len(first_sample['signal_segment'])}")
    if segmented_target_data:
        print("\n--- 目标域分段后首个样本示例 ---")
        first_sample = segmented_target_data[0]
        print(f"源文件: {first_sample['filename']}")
        print(f"传感器: {first_sample['sensor']}")
        print(f"样本长度: {len(first_sample['signal_segment'])}")