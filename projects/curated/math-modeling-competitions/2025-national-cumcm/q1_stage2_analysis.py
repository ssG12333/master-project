import pickle
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import hilbert
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 25
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

tqdm.pandas(desc="提取特征进度")

INPUT_DATA_FILE = 'preprocessed_data.pkl'
N_COMPONENTS = 3
N_CLUSTERS = 5
N_TOP_FEATURES = 30
OUTPUT_DIR = 'plots'
TARGET_FS = 32000
CORRELATION_THRESHOLD = 0.9  # 高度相关的阈值
ALPHA_WEIGHT = 0.6  # 混合评分中，领域差异度的惩罚权重

PARAMS_DE = {'n_balls': 9, 'd_ball': 0.3126, 'd_pitch': 1.537}
PARAMS_FE = {'n_balls': 9, 'd_ball': 0.2656, 'd_pitch': 1.122}
def calculate_fault_frequencies(rpm, sensor):
    """
    根据转速和传感器位置，计算理论故障频率
    """
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
def get_harmonic_energy_ratio(fft_power, fft_freqs, fundamental_freq, num_harmonics=1, bandwidth=5):
    if fundamental_freq == 0: return 0
    total_energy = np.sum(fft_power)
    if total_energy == 0: return 0
    harmonic_energy = 0
    for i in range(1, num_harmonics + 1):
        harmonic_freq = fundamental_freq * i
        freq_indices = np.where(np.abs(fft_freqs - harmonic_freq) <= bandwidth)
        harmonic_energy += np.sum(fft_power[freq_indices])
    return harmonic_energy / total_energy
def extract_features(row):
    segment = np.array(row['signal_segment'])
    rpm = row['rpm']
    sensor = row['sensor']
    features = {}
    features['time_mean'] = np.mean(segment)
    features['time_std'] = np.std(segment)
    features['time_var'] = np.var(segment)
    features['time_rms'] = np.sqrt(np.mean(segment ** 2))
    features['time_peak'] = np.max(np.abs(segment))
    features['time_min'] = np.min(segment)
    features['time_max'] = np.max(segment)
    features['time_p2p'] = features['time_max'] - features['time_min']
    features['time_skew'] = stats.skew(segment)
    features['time_kurtosis'] = stats.kurtosis(segment)
    features['time_abs_mean'] = np.mean(np.abs(segment))
    features['time_energy'] = np.sum(segment ** 2)
    features['crest_factor'] = features['time_peak'] / features['time_rms'] if features['time_rms'] != 0 else 0
    features['shape_factor'] = features['time_rms'] / features['time_abs_mean'] if features['time_abs_mean'] != 0 else 0
    features['impulse_factor'] = features['time_peak'] / features['time_abs_mean'] if features['time_abs_mean'] != 0 else 0
    features['clearance_factor'] = features['time_peak'] / (np.mean(np.sqrt(np.abs(segment))) ** 2) if np.mean(np.sqrt(np.abs(segment))) != 0 else 0
    hist, _ = np.histogram(segment, bins=10, density=True)
    features['shannon_entropy'] = stats.entropy(hist)
    features['zero_crossing_rate'] = ((segment[:-1] * segment[1:]) < 0).sum() / (len(segment) - 1)
    N = len(segment)
    fft_coeffs = np.fft.fft(segment)[:N // 2]
    fft_power = np.abs(fft_coeffs) ** 2 / N
    fft_freqs = np.fft.fftfreq(N, 1 / TARGET_FS)[:N // 2]
    features['freq_mean'] = np.average(fft_freqs, weights=fft_power) if np.sum(fft_power) > 0 else 0
    features['freq_std'] = np.sqrt(np.average((fft_freqs - features['freq_mean']) ** 2, weights=fft_power)) if np.sum(fft_power) > 0 else 0
    features['freq_skew'] = stats.skew(fft_power)
    features['freq_kurtosis'] = stats.kurtosis(fft_power)
    power_norm = fft_power / np.sum(fft_power) if np.sum(fft_power) > 0 else fft_power
    features['freq_entropy'] = stats.entropy(power_norm)
    band_width = 1600
    for i in range(10):
        start_freq = i * band_width
        end_freq = (i + 1) * band_width
        band_indices = np.where((fft_freqs >= start_freq) & (fft_freqs < end_freq))
        features[f'band_energy_{i}'] = np.sum(fft_power[band_indices])
    wp = pywt.WaveletPacket(data=segment, wavelet='db4', mode='symmetric', maxlevel=3)
    nodes = wp.get_level(3, order='natural')
    for i, node in enumerate(nodes):
        coeffs = node.data
        features[f'wpt_energy_{i}'] = np.sum(coeffs ** 2)
        features[f'wpt_std_{i}'] = np.std(coeffs)
    features['quantile_25'] = np.quantile(segment, 0.25)
    features['quantile_75'] = np.quantile(segment, 0.75)
    features['iqr'] = features['quantile_75'] - features['quantile_25']
    features['median_abs_dev'] = stats.median_abs_deviation(segment)
    threshold = features['time_mean'] + 3 * features['time_std']
    features['peak_count'] = (segment > threshold).sum()
    autocorr = np.correlate(segment, segment, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    features['autocorr_lag1'] = autocorr[1] / autocorr[0] if autocorr[0] != 0 else 0
    zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
    features['autocorr_first_zero'] = zero_crossings[0] if len(zero_crossings) > 0 else -1
    analytic_signal = hilbert(segment)
    envelope = np.abs(analytic_signal)
    features['env_mean'] = np.mean(envelope)
    features['env_std'] = np.std(envelope)
    features['env_rms'] = np.sqrt(np.mean(envelope ** 2))
    features['env_skew'] = stats.skew(envelope)
    features['env_kurtosis'] = stats.kurtosis(envelope)
    diff_segment = np.diff(segment)
    features['diff_mean'] = np.mean(diff_segment)
    features['diff_std'] = np.std(diff_segment)
    features['diff_rms'] = np.sqrt(np.mean(diff_segment ** 2))
    features['diff_skew'] = stats.skew(diff_segment)
    features['diff_kurtosis'] = stats.kurtosis(diff_segment)
    if 'sensor' in row and row['sensor'] not in ['target', 'BA']:
        fault_freqs = calculate_fault_frequencies(row.get('rpm'), row['sensor'])
        for fault_name, freq_val in fault_freqs.items():
            for i in range(1, 4):
                features[f'{fault_name}_{i}x_ratio'] = get_harmonic_energy_ratio(
                    fft_power, fft_freqs, freq_val * i, num_harmonics=1, bandwidth=5)
    else:
        for fault_name in ['BPFO', 'BPFI', 'BSF']:
            for i in range(1, 4):
                features[f'{fault_name}_{i}x_ratio'] = 0

    return pd.Series(features)
def plot_correlation_heatmap(df, feature_columns, title, filename):
    print(f"正在生成热力图: {title}...")
    plt.figure(figsize=(28, 24))
    source_only_df = df[df['domain'] == 'source'][feature_columns].dropna()
    correlation_matrix = source_only_df.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
    plt.title(title, fontsize=28)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout(pad=2.0)
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"热力图已保存至: {save_path}")
def remove_highly_correlated_features(df, feature_columns, threshold):
    print(f"\n--- 步骤 2.1: 剔除高度相关的特征 (阈值 > {threshold}) ---")
    source_features_df = df[df['domain'] == 'source'][feature_columns].dropna()
    corr_matrix = source_features_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"找到 {len(to_drop)} 个高度相关的特征将被移除。")
    if to_drop: print("待移除列表:", to_drop)
    features_to_keep = [col for col in feature_columns if col not in to_drop]
    return features_to_keep
def select_transferable_features(df, feature_columns, alpha):
    print(f"\n--- 步骤 2.2: 使用混合评分机制进行特征精选 ---")
    print(f"惩罚权重 alpha = {alpha}")
    print("  正在计算特征的“区分度分数” (RF Importance)...")
    source_df = df[df['domain'] == 'source'].dropna(subset=['fault_type'])
    X_source = source_df[feature_columns].fillna(0)
    y_source = source_df['fault_type']
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_source, y_source)
    discriminative_scores = pd.Series(rf.feature_importances_, index=feature_columns, name='discriminative')
    print("  正在计算特征的“领域差异度分数”...")
    domain_scores = {}
    y_domain = (df['domain'] == 'target').astype(int)
    for feature in tqdm(feature_columns, desc="评估领域差异"):
        X_domain = df[[feature]].fillna(0)
        lr = LogisticRegression(random_state=42, class_weight='balanced')
        accuracy = np.mean(cross_val_score(lr, X_domain, y_domain, cv=3, scoring='accuracy'))
        difference_score = (accuracy - 0.5) * 2
        domain_scores[feature] = max(0, difference_score)
    domain_scores = pd.Series(domain_scores, name='domain_diff')
    scores_df = pd.concat([discriminative_scores, domain_scores], axis=1)
    scaler = MinMaxScaler()
    scores_df[['discriminative_norm', 'domain_diff_norm']] = scaler.fit_transform(scores_df)
    scores_df['final_score'] = scores_df['discriminative_norm'] - alpha * scores_df['domain_diff_norm']
    scores_df = scores_df.sort_values(by='final_score', ascending=False)
    print("\n--- 混合评分最高的 Top 15 特征 ---")
    print(scores_df.head(15))
    top_features_df_composition = scores_df.head(N_TOP_FEATURES)
    plt.figure(figsize=(18, 14))  # 调整图形大小
    top_features_df_composition[['discriminative_norm', 'domain_diff_norm']].plot(
        kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'], figsize=(18, 14), fontsize=14
    )
    plt.title(f'Top {N_TOP_FEATURES} 可迁移特征的混合评分构成', fontsize=24)
    plt.ylabel('归一化分数', fontsize=20)
    plt.xlabel('特征名称', fontsize=20)
    plt.xticks(rotation=90)
    plt.legend(['区分度 (越高越好)', '领域差异度 (越低越好)'], fontsize=18)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(OUTPUT_DIR, "mixed_score_feature_selection.png"))
    plt.close()
    print("混合评分构成图已保存。")
    top_features_df_ranking = scores_df.head(N_TOP_FEATURES).sort_values('final_score', ascending=True)
    plt.figure(figsize=(18, 14))  # 调整图形大小与上面一致
    plt.barh(top_features_df_ranking.index, top_features_df_ranking['final_score'], color='c')
    plt.xlabel('最终混合评分 (越高越好)', fontsize=20)
    plt.ylabel('特征名称', fontsize=20)
    plt.title(f'Top {N_TOP_FEATURES} 可迁移特征最终重要性排序', fontsize=24)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(OUTPUT_DIR, "final_feature_importance_ranking.png"))
    plt.close()
    print("最终特征重要性排序图已保存。")
    final_selected_features = scores_df.head(N_TOP_FEATURES).index.tolist()
    print(f"\n已根据混合评分选取 {len(final_selected_features)} 个特征。")
    return final_selected_features
def plot_pca_and_clustering(full_df_with_features, final_selected_features):
    print("\n--- 步骤 3: 使用最终特征进行PCA和聚类 ---")
    features_to_scale = full_df_with_features[final_selected_features].fillna(0)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_to_scale)
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca_result = pca.fit_transform(features_scaled)
    explained_variance = pca.explained_variance_ratio_
    print(f"\nPCA 主成分解释方差比例: {explained_variance}")
    print(f"前 2 个主成分累计解释方差: {np.sum(explained_variance[:2]):.2%}")
    print(f"前 3 个主成分累计解释方差: {np.sum(explained_variance):.2%}")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    full_df_with_features['cluster'] = cluster_labels
    is_target_mask = (full_df_with_features['domain'] == 'target').values
    fault_type_labels = full_df_with_features['fault_type'].values
    fault_types = sorted(np.unique(fault_type_labels[~is_target_mask]))
    colors = plt.colormaps.get('tab10')
    fig_3d_pca = plt.figure(figsize=(18, 14))
    ax_3d_pca = fig_3d_pca.add_subplot(111, projection='3d')
    for i, ft in enumerate(fault_types):
        idx = np.where((fault_type_labels == ft) & (~is_target_mask))
        ax_3d_pca.scatter(pca_result[idx, 0], pca_result[idx, 1], pca_result[idx, 2], label=f'源域 - {ft}', c=[colors(i)], alpha=0.5, s=40)
    target_idx = np.where(is_target_mask)
    ax_3d_pca.scatter(pca_result[target_idx, 0], pca_result[target_idx, 1], pca_result[target_idx, 2], label='目标域', c='red', marker='x', s=60, alpha=0.9)
    ax_3d_pca.set_title(f'3D PCA 特征空间分布 (累计解释方差: {np.sum(explained_variance):.2%})', fontsize=24)
    ax_3d_pca.legend(fontsize=18)
    ax_3d_pca.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_distribution_3d.png"))
    plt.close(fig_3d_pca)
    plt.figure(figsize=(16, 12))
    for i, ft in enumerate(fault_types):
        idx = np.where((fault_type_labels == ft) & (~is_target_mask))
        plt.scatter(pca_result[idx, 0], pca_result[idx, 1], label=f'源域 - {ft}', c=[colors(i)], alpha=0.5, s=40)
    plt.scatter(pca_result[target_idx, 0], pca_result[target_idx, 1], label='目标域', c='red', marker='x', s=60, alpha=0.9)
    plt.title(f'2D PCA 特征空间分布 (累计解释方差: {np.sum(explained_variance[:2]):.2%})', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_distribution_2d.png"))
    plt.close()
    print("2D 和 3D PCA分布图已保存。")
    unique_clusters = np.unique(cluster_labels)
    cluster_colors = plt.colormaps.get('tab20')
    fig_3d_cluster = plt.figure(figsize=(18, 14))
    ax_3d_cluster = fig_3d_cluster.add_subplot(111, projection='3d')
    ax_3d_cluster.scatter(pca_result[~is_target_mask, 0], pca_result[~is_target_mask, 1], pca_result[~is_target_mask, 2], c=cluster_labels[~is_target_mask], cmap=cluster_colors, alpha=0.4, s=40)
    ax_3d_cluster.scatter(pca_result[is_target_mask, 0], pca_result[is_target_mask, 1], pca_result[is_target_mask, 2], c=cluster_labels[is_target_mask], cmap=cluster_colors, marker='x', s=80,
                    alpha=1.0)
    ax_3d_cluster.set_title('3D PCA 空间中的 KMeans 聚类结果', fontsize=24)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'簇 {i}', markerfacecolor=cluster_colors(i / (len(unique_clusters) - 1))) for i in unique_clusters]
    ax_3d_cluster.legend(handles=handles, title="聚类ID", fontsize=18, title_fontsize=20)
    ax_3d_cluster.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(OUTPUT_DIR, "clustering_result_3d.png"))
    plt.close(fig_3d_cluster)
    plt.figure(figsize=(16, 12))
    plt.scatter(pca_result[~is_target_mask, 0], pca_result[~is_target_mask, 1], c=cluster_labels[~is_target_mask], cmap=cluster_colors, alpha=0.4, s=40)
    plt.scatter(pca_result[is_target_mask, 0], pca_result[is_target_mask, 1], c=cluster_labels[is_target_mask], cmap=cluster_colors, marker='x', s=80, alpha=1.0)
    plt.title('2D PCA 空间中的 KMeans 聚类结果', fontsize=24)
    plt.grid(True)
    plt.legend(handles=handles, title="聚类ID", fontsize=18, title_fontsize=20)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(OUTPUT_DIR, "clustering_result_2d.png"))
    plt.close()
    print("2D 和 3D 聚类结果图已保存。")
    return full_df_with_features
if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print(f"正在从 {INPUT_DATA_FILE} 加载数据...")
    with open(INPUT_DATA_FILE, 'rb') as f:
        segmented_source_data, segmented_target_data = pickle.load(f)
    print("数据加载完成。")
    print("\n--- 步骤 1: 特征提取 ---")
    source_df = pd.DataFrame(segmented_source_data)
    target_df = pd.DataFrame(segmented_target_data)
    source_df['domain'] = 'source'
    target_df['domain'] = 'target'
    full_df = pd.concat([source_df, target_df], ignore_index=True)
    features_df = full_df.progress_apply(extract_features, axis=1)
    full_df_with_features = pd.concat([full_df, features_df], axis=1)
    all_feature_columns = features_df.columns.tolist()
    print(f"特征提取完成。共提取 {len(all_feature_columns)} 个特征。")
    print("\n--- 步骤 2: 高级特征选择流程 ---")
    plot_correlation_heatmap(full_df_with_features, all_feature_columns,
                             '全部75个特征相关性热力图 (初筛前)',
                             'correlation_heatmap_all_76_features.png')
    non_redundant_features = remove_highly_correlated_features(
        full_df_with_features, all_feature_columns, threshold=CORRELATION_THRESHOLD
    )
    print(f"移除冗余后，剩余 {len(non_redundant_features)} 个特征用于后续筛选。")
    final_selected_features = select_transferable_features(
        full_df_with_features, non_redundant_features, alpha=ALPHA_WEIGHT
    )

    plot_correlation_heatmap(full_df_with_features, final_selected_features,
                             f'最终筛选的 {len(final_selected_features)} 个可迁移特征相关性热力图 (验证)',
                             'correlation_heatmap_final_selected_features.png')
    with open('selected_features.json', 'w') as f:
        json.dump(final_selected_features, f)
    print("已选择的特征列表已保存至: selected_features.json")
    full_df_with_features = plot_pca_and_clustering(full_df_with_features, final_selected_features)
    print("\n--- 步骤 4: 根据聚类结果筛选源域数据 ---")
    target_clusters = full_df_with_features[full_df_with_features['domain'] == 'target']['cluster'].unique()
    similar_source_indices = full_df_with_features[
        (full_df_with_features['domain'] == 'source') &
        (full_df_with_features['cluster'].isin(target_clusters))
        ].index
    final_source_df = full_df_with_features.loc[similar_source_indices].copy()
    final_target_df = full_df_with_features[full_df_with_features['domain'] == 'target'].copy()
    print(f"筛选后得到的相似源域样本数: {len(final_source_df)}")
    print("\n--- 步骤 5: 保存最终数据集 ---")
    final_source_df.to_parquet('final_source_data.parquet', index=False)
    final_target_df.to_parquet('final_target_data.parquet', index=False)
    print("筛选后的源域和目标域数据集已保存。")
    print("\n第一问第二阶段所有任务已完成！")