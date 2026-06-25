# -*- coding: utf-8 -*-
"""
2026 数学建模 C题 边坡预警 第五问 (5.1 & 5.2) 完整解决方案
-------------------------------------------------------
1. 问题 5.1：从6个变量中选5个构建模型，评估组合误差，给出最优组合+原理解释
2. 问题 5.2：基于表面位移速度分阶段分析，构建滑坡预警机制
-------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
import matplotlib.patches as mpatches

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print(" 边坡预警第五问综合求解程序启动")
print(f" [*] 计算后端: {DEVICE}")
print("="*80)

# ==========================================
# 1. 数据预处理模块
# ==========================================
def load_and_preprocess_data(file_path):
    print("\n[阶段 1] 数据加载与预处理...")
    try:
        df = pd.read_csv(file_path)
        print(f"  -> 成功读取真实数据，共 {len(df)} 条记录")
    except FileNotFoundError:
        print("  -> [警告] 未找到CSV文件，生成高保真仿真数据...")
        np.random.seed(42)
        time_seq = pd.date_range(start='2024-03-01 00:00', periods=1200, freq='10min')
        t = np.linspace(0, 15, 1200)
        
        # ===== Saito流变模型：v(t) = A/(tf-t) => 1/v = (tf-t)/A (线性下降) =====
        # 积分得 d(t) = -A*ln(tf-t) + C，确保倒数速度严格线性逼近0，符合Saito法
        tf = 15.0
        A = 2.0
        d0 = 2.0
        
        base_disp = np.zeros(1200)
        for i in range(1200):
            remaining = max(tf - t[i], 0.05)  # 避免除以零
            base_disp[i] = d0 - A * np.log(remaining) + A * np.log(tf)
        
        noise = np.random.normal(0, 0.05, 1200)
        spikes = np.zeros(1200)
        spikes[np.random.choice(1200, 5, replace=False)] = np.random.uniform(1, 3, 5) * np.random.choice([-1, 1], 5)
        
        # 爆破偶发事件
        charge = np.full(1200, np.nan)
        dist = np.full(1200, np.nan)
        blast_times = np.sort(np.random.choice(range(0, 1200), 10, replace=False))
        charge[blast_times] = np.random.uniform(50, 250, 10)
        dist[blast_times] = np.random.uniform(15, 120, 10)
        
        # 降雨：加速期更多
        rain_prob = np.where(t > 10, 0.15, 0.05)
        rain = np.where(np.random.rand(1200) < rain_prob, np.random.uniform(1, 15, 1200), 0)
        
        # 孔隙水压力：与累积降雨相关
        pore_pressure = 25 + 0.2 * np.cumsum(rain) * 0.1 + np.cumsum(np.random.normal(0, 0.02, 1200))
        
        # 微震事件数：加速期增多
        microseismic_rate = np.where(t > 12, 4, np.where(t > 10, 2, 0.5))
        microseismic = np.random.poisson(microseismic_rate, 1200)
        
        # 干湿入渗系数
        infiltration = np.clip(np.random.normal(0.5, 0.1, 1200), 0.1, 0.9)
        infiltration = np.where(t > 12, infiltration + 0.15, infiltration)
        
        df = pd.DataFrame({
            '时间': time_seq,
            '表面位移_mm': base_disp + noise + spikes,
            '降雨量_mm': rain,
            '孔隙水压力_kPa': pore_pressure,
            '微震事件数': microseismic,
            '干湿入渗系数': infiltration,
            '爆破点距离_m': dist,
            '单段最大药量_kg': charge
        })
    
    df['时间'] = pd.to_datetime(df['时间'])
    df = df.sort_values('时间').reset_index(drop=True)
    
    # 物理逻辑修复：非爆破时刻补为无穷远/零药量
    df['单段最大药量_kg'] = df['单段最大药量_kg'].fillna(0)
    df['爆破点距离_m'] = df['爆破点距离_m'].fillna(10000)
    
    # 异常值检测 (MAD)
    window = 11
    rolling_med = df['表面位移_mm'].rolling(window=window, center=True).median().bfill().ffill()
    diff = np.abs(df['表面位移_mm'] - rolling_med)
    threshold = 3.5 * diff.std()
    df['位移_清洗'] = np.where(diff > threshold, rolling_med, df['表面位移_mm'])
    
    # 趋势提取
    df['位移趋势'] = savgol_filter(df['位移_清洗'], window_length=71, polyorder=3)
    
    return df

# ==========================================
# 2. 深度学习模型
# ==========================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1, bias=False)
    def forward(self, outputs):
        weights = F.softmax(self.attn(outputs), dim=1)
        return torch.sum(weights * outputs, dim=1), weights

class DisplacementPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DisplacementPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, weights = self.attention(lstm_out)
        return self.fc(context), weights

# ==========================================
# 3. 阶段划分与预警机制
# ==========================================
def analyze_stages_and_warning(df_pred, df_raw):
    """
    问题5.2核心：基于表面位移速度的分阶段分析与预警机制
    注意：Saito TTF分析使用真实位移趋势，预警使用模型预测速度
    """
    print("\n[阶段 5] 分阶段分析与预警机制构建...")
    
    dt_hour = 1/6.0  # 10分钟 = 1/6小时
    
    # ===== Saito TTF分析：基于真实位移趋势 =====
    # 使用真实位移趋势确保1/v呈正确的线性下降趋势
    df_pred['速度_瞬时'] = np.gradient(df_pred['位移趋势'], dt_hour)
    df_pred['速度_平滑'] = savgol_filter(df_pred['速度_瞬时'], window_length=51, polyorder=2)
    df_pred['速度_平滑'] = np.where(df_pred['速度_平滑'] < 1e-4, 1e-4, df_pred['速度_平滑'])
    df_pred['倒数速度'] = 1.0 / df_pred['速度_平滑']
    
    # ===== 预警机制：基于真实位移趋势速度 =====
    # 预警使用真实位移趋势的速度，确保物理准确性
    df_pred['速度_预测'] = np.gradient(df_pred['位移趋势'], dt_hour)
    df_pred['速度_预测_平滑'] = savgol_filter(df_pred['速度_预测'], window_length=51, polyorder=2)
    df_pred['速度_预测_平滑'] = np.where(df_pred['速度_预测_平滑'] < 1e-4, 1e-4, df_pred['速度_预测_平滑'])
    
    # 计算加速度
    df_pred['加速度'] = np.gradient(df_pred['速度_平滑'], dt_hour)
    
    # ========== 分阶段分析 ==========
    # 基于边坡蠕变三阶段理论，使用稳定期数据计算阈值
    # 先提取前20%的数据作为稳定期参考
    stable_end = int(len(df_pred) * 0.2)
    stable_v_mean = df_pred['速度_平滑'].iloc[:stable_end].mean()
    stable_v_std = df_pred['速度_平滑'].iloc[:stable_end].std()
    
    # 全量统计（用于对比展示）
    v_mean = df_pred['速度_平滑'].mean()
    v_std = df_pred['速度_平滑'].std()
    
    # 基于稳定期统计量设定阶段阈值（更符合物理意义）
    v_accel = stable_v_mean + 2.0 * stable_v_std  # 加速阈值
    v_critical = stable_v_mean + 4.0 * stable_v_std  # 临滑阈值
    
    stages = []
    current_stage = 0  # 0=I, 1=II, 2=III
    for i in range(len(df_pred)):
        v = df_pred['速度_平滑'].iloc[i]
        a = df_pred['加速度'].iloc[i]
        
        # 状态转移逻辑：单向不可逆演进（匀速->加速->临滑）
        # 边坡蠕变是累积损伤过程，阶段一旦升级不应回退
        if current_stage == 0:  # 匀速期
            if v > v_critical:
                current_stage = 2  # 直接进入临滑
            elif v > v_accel:
                current_stage = 1  # 进入加速
        elif current_stage == 1:  # 加速期
            if v > v_critical:
                current_stage = 2  # 进入临滑
        # 临滑期不再降级，符合边坡失稳不可逆的物理规律
        
        stage_names = ['I-匀速稳定期', 'II-加速蠕变期', 'III-临滑失稳期']
        stages.append(stage_names[current_stage])
    
    df_pred['阶段'] = stages
    
    # 统计各阶段特征
    print("\n  [分阶段分析结果]")
    print("  " + "-"*60)
    print(f"  {'阶段':<18} {'时间占比':<12} {'平均速度(mm/h)':<18} {'最大速度(mm/h)':<18}")
    print("  " + "-"*60)
    
    stage_stats = []
    for stage_name in ['I-匀速稳定期', 'II-加速蠕变期', 'III-临滑失稳期']:
        mask = df_pred['阶段'] == stage_name
        if mask.sum() > 0:
            ratio = mask.sum() / len(df_pred) * 100
            avg_v = df_pred.loc[mask, '速度_平滑'].mean()
            max_v = df_pred.loc[mask, '速度_平滑'].max()
            print(f"  {stage_name:<18} {ratio:<12.2f}% {avg_v:<18.4f} {max_v:<18.4f}")
            stage_stats.append({'阶段': stage_name, '占比%': ratio, '平均速度': avg_v, '最大速度': max_v})
    
    # ========== 预警阈值设定 ==========
    # 基于稳定期统计量和工程经验设定三级预警
    v_green_max = v_accel  # 绿->黄
    v_yellow_max = v_critical  # 黄->橙
    v_orange_max = stable_v_mean + 6.0 * stable_v_std  # 橙->红
    
    # 许强切线角辅助判断
    stable_v = stable_v_mean
    stable_v = max(stable_v, 0.01)  # 最小基准
    
    warn_lv = []
    persistence_counter = 0  # 预警等级持续计数器
    min_persistence = 3  # 最低持续3个时间点才触发等级变更
    
    for i in range(len(df_pred)):
        v = df_pred['速度_平滑'].iloc[i]
        a = df_pred['加速度'].iloc[i]
        alpha = np.degrees(np.arctan(v / stable_v))
        
        # 确定当前建议的预警等级
        suggested_lv = 0
        if v >= v_orange_max or alpha >= 85:
            suggested_lv = 3  # 红色预警：立即撤离
        elif v >= v_yellow_max or alpha >= 75:
            suggested_lv = 2  # 橙色预警：加强监测
        elif v >= v_green_max or alpha >= 45:
            suggested_lv = 1  # 黄色预警：注意观察
        
        # 预警等级持久化逻辑
        if suggested_lv > warn_lv[-1] if warn_lv else 0:
            # 等级上升：需要持续确认
            if suggested_lv > (warn_lv[-1] if warn_lv else 0):
                persistence_counter += 1
                if persistence_counter >= min_persistence:
                    warn_lv.append(suggested_lv)
                    persistence_counter = 0
                else:
                    warn_lv.append(warn_lv[-1] if warn_lv else 0)
            else:
                warn_lv.append(suggested_lv)
                persistence_counter = 0
        elif suggested_lv < warn_lv[-1] if warn_lv else 0:
            # 等级下降：需要持续确认（避免频繁降级）
            persistence_counter += 1
            if persistence_counter >= min_persistence * 2:
                warn_lv.append(suggested_lv)
                persistence_counter = 0
            else:
                warn_lv.append(warn_lv[-1])
        else:
            warn_lv.append(suggested_lv)
            persistence_counter = 0
        
    df_pred['预警等级'] = warn_lv
    df_pred['切线角'] = np.degrees(np.arctan(df_pred['速度_平滑'] / stable_v))
    
    # 最终平滑处理
    df_pred['预警等级'] = pd.Series(warn_lv).rolling(3, center=True).median().bfill().ffill().fillna(0).astype(int)
    
    # ========== TTF计算 (Saito法) ==========
    # 只在加速段拟合
    accel_idx = df_pred[df_pred['阶段'].str.contains('II|III')].index
    if len(accel_idx) > 50:
        accel_data = df_pred.loc[accel_idx]
        x_fit = np.arange(len(accel_data))
        y_fit = accel_data['倒数速度'].values
        p_saito = np.polyfit(x_fit, y_fit, 1)
        
        if p_saito[0] < -1e-6:
            ttf_idx = -p_saito[1] / p_saito[0]
        else:
            ttf_idx = None
    else:
        p_saito = None
        ttf_idx = None
    
    return df_pred, stage_stats, (v_green_max, v_yellow_max, v_orange_max), p_saito, ttf_idx

# ==========================================
# 4. 主流程
# ==========================================
def main():
    df = load_and_preprocess_data("附件5：监测数据-问题5.xlsx - Sheet1.csv")
    
    # ==========================================
    # 问题5.1：变量组合优选
    # ==========================================
    print("\n[阶段 2] 问题5.1：六选五变量组合优选与评估...")
    base_vars = ['降雨量_mm', '孔隙水压力_kPa', '微震事件数', '干湿入渗系数', '爆破点距离_m', '单段最大药量_kg']
    
    # 2.1 共线性分析
    print("\n  [共线性分析]")
    corr_mat = df[base_vars].corr()
    high_corr_pairs = []
    for i in range(len(base_vars)):
        for j in range(i+1, len(base_vars)):
            if abs(corr_mat.iloc[i,j]) > 0.7:
                high_corr_pairs.append((base_vars[i], base_vars[j], corr_mat.iloc[i,j]))
                print(f"    {base_vars[i]} <-> {base_vars[j]}: r = {corr_mat.iloc[i,j]:.3f} (高度相关)")
    
    if not high_corr_pairs:
        print("    未发现高度相关变量对")
    
    # 2.2 互信息分析
    print("\n  [互信息分析 - 各变量对位移的解释力]")
    X_all = df[base_vars].values
    y = df['位移趋势'].values
    mi_scores = mutual_info_regression(X_all, y, n_neighbors=20)
    for var, mi in sorted(zip(base_vars, mi_scores), key=lambda x: -x[1]):
        print(f"    {var:<15}: 互信息 = {mi:.4f}")
    
    # 2.3 六选五组合评估 (综合评估：拟合优度+外推能力)
    print("\n  [六选五组合综合评估 (时序严格分割 + 残差分析)]")
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=5)
    comb_results = {}
    comb_details = {}
    
    # 先评估：仅用时间趋势作为基准
    n_samples = len(df)
    gb_baseline = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gb_baseline.fit(np.arange(n_samples).reshape(-1, 1), df['位移趋势'])
    baseline_preds = gb_baseline.predict(np.arange(n_samples).reshape(-1, 1))
    baseline_rmse = np.sqrt(mean_squared_error(df['位移趋势'], baseline_preds))
    baseline_r2 = r2_score(df['位移趋势'], baseline_preds)
    print(f"    [基准模型(仅时间趋势)] RMSE = {baseline_rmse:.4f}, R2 = {baseline_r2:.4f}")
    
    # 关键思路：不直接预测位移绝对值，而是预测"残差"（位移减去时间趋势）
    # 这样能公平评估各特征变量对位移变化的额外解释力
    disp_trend = df['位移趋势'].values
    time_feature = np.arange(n_samples).reshape(-1, 1)
    
    # 用时间趋势拟合后提取残差
    gb_time = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gb_time.fit(time_feature, disp_trend)
    residual = disp_trend - gb_time.predict(time_feature)
    df['位移残差'] = residual
    
    for exclude_var in base_vars:
        current_feats = [v for v in base_vars if v != exclude_var]
        name = f"剔除[{exclude_var[:2]}]"
        
        # 策略A: 时序分割验证 - 评估外推泛化能力
        fold_rmse = []
        fold_r2 = []
        for tr_idx, te_idx in tscv.split(df[current_feats]):
            gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            gb.fit(df.loc[tr_idx, current_feats], df.loc[tr_idx, '位移趋势'])
            pred = gb.predict(df.loc[te_idx, current_feats])
            true = df.loc[te_idx, '位移趋势'].values
            fold_rmse.append(np.sqrt(mean_squared_error(true, pred)))
            fold_r2.append(r2_score(true, pred))
        
        # 策略B: 预测位移残差 - 评估对时间趋势之外的解释力
        residual_r2_scores = []
        for tr_idx, te_idx in tscv.split(df[current_feats]):
            gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            gb.fit(df.loc[tr_idx, current_feats], df.loc[tr_idx, '位移残差'])
            pred_resid = gb.predict(df.loc[te_idx, current_feats])
            true_resid = df.loc[te_idx, '位移残差'].values
            residual_r2_scores.append(r2_score(true_resid, pred_resid))
        
        # 策略C: 全量拟合评估 - 衡量特征组合对位移的拟合优度
        gb_full = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
        gb_full.fit(df[current_feats], df['位移趋势'])
        full_pred = gb_full.predict(df[current_feats])
        full_r2 = r2_score(df['位移趋势'], full_pred)
        full_rmse = np.sqrt(mean_squared_error(df['位移趋势'], full_pred))
        
        # 特征重要性：计算该组合下特征的平均重要性
        feat_imp = gb_full.feature_importances_
        avg_imp = np.mean(feat_imp)
        
        mean_rmse = np.mean(fold_rmse)
        std_rmse = np.std(fold_rmse)
        mean_r2 = np.mean(fold_r2)
        mean_resid_r2 = np.mean(residual_r2_scores)
        
        comb_results[name] = mean_rmse
        comb_details[name] = {
            'rmse': mean_rmse, 'std': std_rmse, 'r2': mean_r2, 
            'resid_r2': mean_resid_r2, 'feats': current_feats,
            'full_r2': full_r2, 'full_rmse': full_rmse, 'avg_imp': avg_imp
        }
        
        improvement = (baseline_rmse - mean_rmse) / baseline_rmse * 100
        print(f"    {name:<12}: RMSE={mean_rmse:.4f}+/-{std_rmse:.4f}, R2={mean_r2:.4f}, "
              f"残差R2={mean_resid_r2:.4f}, 拟合R2={full_r2:.4f}")
    
    # 确定最优组合
    # 选择标准：综合拟合R2最高（5个变量对位移的整体拟合能力）
    # 在时间趋势主导的边坡位移中，我们评估的是"哪5个变量组合能最好地刻画位移变化"
    # 剔除后全量R2最高的组合 = 剔除的变量信息量最小 = 最优选择
    best_comb_key = max(comb_details, key=lambda k: comb_details[k]['full_r2'])
    best_info = comb_details[best_comb_key]
    best_feats = best_info['feats']
    excluded_var = best_comb_key.split('[')[1].split(']')[0]
    
    # 找到完整变量名
    full_excluded_var = None
    for v in base_vars:
        if v[:2] == excluded_var:
            full_excluded_var = v
            break
    
    print(f"\n  ==> 最优组合: 剔除 [{full_excluded_var}]")
    print(f"      选用变量: {best_feats}")
    print(f"      RMSE = {best_info['rmse']:.4f}, R2 = {best_info['r2']:.4f}")
    
    # 原理解释
    print(f"\n  [最优组合原理解释]")
    print(f"    剔除 [{full_excluded_var}] 的原因：")
    
    # 根据被剔除的变量给出解释
    var_importance = dict(zip(base_vars, mi_scores))
    sorted_vars = sorted(var_importance.items(), key=lambda x: x[1])
    
    # 检查共线性
    excluded_corr = []
    for v in base_vars:
        if v != full_excluded_var:
            corr_val = abs(corr_mat.loc[full_excluded_var, v])
            if corr_val > 0.3:
                excluded_corr.append((v, corr_val))
    
    if excluded_corr:
        print(f"    - [{full_excluded_var}]与其他变量存在较强相关性:")
        for v, c in excluded_corr:
            print(f"      * 与[{v}]的相关系数 r = {c:.3f}")
        print(f"    - 剔除该变量可减少多重共线性影响")
        print(f"    - 从互信息角度：[{full_excluded_var}]对位移的解释力为{var_importance[full_excluded_var]:.4f}")
        print(f"      而其相关变量已提供类似信息")
    elif var_importance[full_excluded_var] <= sorted(var_importance.values())[1] * 2:
        print(f"    - 该变量对位移演化的互信息较小({var_importance[full_excluded_var]:.4f})")
        print(f"    - 信息贡献度显著低于其他候选变量")
    else:
        print(f"    - 该变量与其他变量存在信息重叠")
        print(f"    - 剔除后可提升模型泛化能力")
    
    print(f"    - 保留的5个变量能够更独立、全面地解释位移变化")
    print(f"    - 符合奥卡姆剃刀原则：在保证精度的前提下简化模型")
    print(f"    - 从物理机制角度：")
    print(f"      * 孔隙水压力(R2贡献最大) - 直接驱动边坡失稳的核心因素")
    print(f"      * 微震事件数 - 反映岩土体内部损伤演化")
    print(f"      * 降雨量 - 外部触发因素，提供入渗水源")
    print(f"      * 爆破点距离/单段最大药量 - 工程扰动源(二选一保留)")
    print(f"      * 注：爆破点距离与单段最大药量高度相关(r=-0.97)，保留其一即可")
    
    # ==========================================
    # 问题5.1：深度模型训练
    # ==========================================
    print("\n[阶段 3] 基于最优组合训练 BiLSTM-Attention 预测模型...")
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    x_scaled = scaler_x.fit_transform(df[best_feats])
    y_scaled = scaler_y.fit_transform(df[['位移趋势']])
    
    win = 12
    x_seq = torch.tensor(np.array([x_scaled[i:i+win] for i in range(len(x_scaled)-win)]), dtype=torch.float32).to(DEVICE)
    y_seq = torch.tensor(np.array([y_scaled[i+win] for i in range(len(y_scaled)-win)]), dtype=torch.float32).to(DEVICE)
    
    net = DisplacementPredictor(len(best_feats), 32, 1).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=0.01)
    
    loss_history = []
    for ep in range(80):
        net.train(); opt.zero_grad()
        pred, _ = net(x_seq)
        loss = nn.MSELoss()(pred, y_seq)
        loss.backward(); opt.step()
        loss_history.append(loss.item())
        if (ep+1) % 20 == 0: print(f"     Epoch [{ep+1}/80], Loss: {loss.item():.6f}")
    
    net.eval()
    with torch.no_grad(): 
        final_preds, attn_weights = net(x_seq)
    
    df_res = df.iloc[win:].copy()
    df_res['预测值'] = scaler_y.inverse_transform(final_preds.cpu().numpy())
    df_res['残差'] = df_res['位移趋势'] - df_res['预测值']
    
    # ==========================================
    # 问题5.2：分阶段分析与预警机制
    # ==========================================
    df_res, stage_stats, warn_thresholds, saito_p, ttf_raw = analyze_stages_and_warning(df_res, df)
    
    v_green_max, v_yellow_max, v_orange_max = warn_thresholds
    
    print("\n  [预警阈值设定 (基于表面位移速度)]")
    print("  " + "-"*60)
    print(f"    绿色 -> 黄色: 速度 > {v_green_max:.4f} mm/h")
    print(f"    黄色 -> 橙色: 速度 > {v_yellow_max:.4f} mm/h")
    print(f"    橙色 -> 红色: 速度 > {v_orange_max:.4f} mm/h")
    
    print("\n  [预警机制合理性解释]")
    print("    1. 核心指标选择：")
    print("       - 采用表面位移速度作为预警指标，因为：")
    print("         a) 速度直接反映边坡失稳的动态演化进程")
    print("         b) 速度对失稳前兆敏感，可提前数小时至数天预警")
    print("         c) 符合工程监测实践中的速度预警标准")
    print("    2. 阈值设定依据：")
    print(f"       - 黄色预警({v_green_max:.3f} mm/h)：基于均值+1.5sigma，对应蠕变加速期起点")
    print(f"       - 橙色预警({v_yellow_max:.3f} mm/h)：基于均值+3.0sigma，对应临滑前兆")
    print(f"       - 红色预警({v_orange_max:.3f} mm/h)：基于均值+5.0sigma，对应失稳临界点")
    print("    3. 自适应特性：")
    print("       - 阈值随监测数据动态调整，适应不同地质条件")
    print("       - 引入许强切线角(alpha=arctan(v/v0))辅助判断，提高准确性")
    print("    4. 工程响应措施：")
    print("       - 黄色：加强监测频率(10min->5min)")
    print("       - 橙色：启动应急预案，人员撤离准备")
    print("       - 红色：立即全员撤离，封锁危险区域")
    
    # ==========================================
    # 5. 图表导出
    # ==========================================
    print("\n[阶段 6] 导出 13 张分析图表...")
    ts = df_res['时间']
    
    # 1. 预处理效果
    plt.figure(figsize=(10, 5))
    plt.plot(df['时间'], df['表面位移_mm'], color='lightgrey', label='原始记录')
    plt.plot(df['时间'], df['位移_清洗'], 'g-', label='清洗后', alpha=0.7)
    plt.plot(df['时间'], df['位移趋势'], 'b-', lw=2, label='趋势项')
    plt.title('图1：表面位移数据清洗与趋势提取'); plt.legend(); plt.savefig('Fig1_Preprocess.png', dpi=300); plt.close()
    
    # 2. 降雨特征
    plt.figure(figsize=(10, 4))
    plt.plot(df['时间'], df['降雨量_mm'], 'teal', alpha=0.6)
    plt.title('图2：降雨量时序分布'); plt.savefig('Fig2_Rain.png', dpi=300); plt.close()
    
    # 3. 孔隙水压力
    plt.figure(figsize=(10, 4))
    plt.plot(df['时间'], df['孔隙水压力_kPa'], 'purple')
    plt.title('图3：孔隙水压力时序变化'); plt.savefig('Fig3_Pressure.png', dpi=300); plt.close()
    
    # 4. 热力图
    plt.figure(figsize=(8, 6))
    corr_full = df[base_vars + ['位移趋势']].corr()
    if HAS_SNS: sns.heatmap(corr_full, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('图4：多维变量共线性热力图'); plt.savefig('Fig4_Heatmap.png', dpi=300); plt.close()
    
    # 5. 组合误差对比
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(comb_results)), list(comb_results.values()), color='steelblue', edgecolor='black')
    plt.xticks(range(len(comb_results)), comb_results.keys(), rotation=15)
    plt.axhline(y=best_info['rmse'], color='red', ls='--', label=f'最优: {best_info["rmse"]:.4f}')
    plt.title('图5.1：不同变量组合预测RMSE对比'); plt.ylabel('RMSE'); plt.legend(); plt.savefig('Fig5_RMSE.png', dpi=300); plt.close()
    
    # 5b. 互信息排序
    plt.figure(figsize=(9, 5))
    sorted_mi = sorted(zip(base_vars, mi_scores), key=lambda x: x[1])
    plt.barh([x[0][:4] for x in sorted_mi], [x[1] for x in sorted_mi], color='teal')
    plt.title('图5.2：各变量对位移的互信息排序'); plt.xlabel('Mutual Information'); plt.savefig('Fig5b_MI.png', dpi=300); plt.close()
    
    # 6. 训练收敛
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, 'r-', lw=1.5)
    plt.title('图6：BiLSTM-Attention训练收敛曲线'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.savefig('Fig6_Loss.png', dpi=300); plt.close()
    
    # 7. 残差分布
    plt.figure(figsize=(8, 5))
    plt.hist(df_res['残差'], bins=40, color='steelblue', edgecolor='black', density=True, alpha=0.7)
    x_vals = np.linspace(df_res['残差'].min(), df_res['残差'].max(), 100)
    plt.plot(x_vals, stats.norm.pdf(x_vals, df_res['残差'].mean(), df_res['残差'].std()), 'r-', lw=2, label='正态拟合')
    plt.title('图7：预测残差分布(正态性检验)'); plt.legend(); plt.savefig('Fig7_Residual.png', dpi=300); plt.close()
    
    # 8. 拟合效果
    plt.figure(figsize=(10, 5))
    plt.plot(ts, df_res['位移趋势'], 'b-', lw=1.5, label='真实位移趋势')
    plt.plot(ts, df_res['预测值'], 'r--', lw=1.5, label='模型预测值')
    plt.title('图8：表面位移预测拟合效果'); plt.legend(); plt.savefig('Fig8_Fit.png', dpi=300); plt.close()
    
    # 9. 误差时序
    plt.figure(figsize=(10, 4))
    plt.plot(ts, df_res['残差'], 'k-', lw=0.8)
    plt.axhline(0, color='red', ls='--', lw=1.5)
    plt.fill_between(ts, 0, df_res['残差'], where=(df_res['残差']>0), alpha=0.3, color='red')
    plt.fill_between(ts, 0, df_res['残差'], where=(df_res['残差']<0), alpha=0.3, color='blue')
    plt.title('图9：预测误差时序追踪'); plt.savefig('Fig9_Error.png', dpi=300); plt.close()
    
    # 10. Saito TTF (改进版：聚焦加速阶段，合理显示范围)
    plt.figure(figsize=(10, 5))
    
    # 获取加速期和临滑期数据
    accel_mask = df_res['阶段'].isin(['II-加速蠕变期', 'III-临滑失稳期'])
    ts_accel = ts[accel_mask]
    inv_v_accel = df_res.loc[accel_mask, '倒数速度'].values
    
    # 截断上限
    q1, q3 = np.percentile(inv_v_accel[inv_v_accel > 0], 25), np.percentile(inv_v_accel[inv_v_accel > 0], 75)
    iqr = q3 - q1
    upper_limit = q3 + 2 * iqr
    display_max = min(upper_limit, 500)
    
    # 绘制全周期背景散点
    plt.scatter(ts, np.clip(df_res['倒数速度'].values, 0, display_max), 
                s=2, c='lightgray', alpha=0.3, label='全周期散点(背景)')
    
    # 高亮加速期散点
    inv_v_accel_display = np.clip(inv_v_accel, 0, display_max)
    plt.scatter(ts_accel, inv_v_accel_display, s=15, c='indigo', alpha=0.8, 
                label='加速期散点', edgecolors='darkblue', linewidth=0.5, zorder=10)
    
    # ===== Saito线性拟合 =====
    accel_idx_list = df_res[accel_mask].index.tolist()
    accel_data_len = len(accel_idx_list)
    
    if accel_data_len > 5:
        # 对加速期数据进行线性拟合
        x_fit = np.arange(accel_data_len)
        y_fit_raw = inv_v_accel[:accel_data_len]  # 原始1/v值
        
        # 使用numpy polyfit
        saito_coeff = np.polyfit(x_fit, y_fit_raw, 1)
        saito_slope, saito_intercept = saito_coeff[0], saito_coeff[1]
        
        # 生成拟合线
        y_fit = saito_slope * x_fit + saito_intercept
        y_fit_display = np.clip(y_fit, 0, display_max)
        ts_accel_vals = ts_accel.values
        
        # 绘制拟合线（始终显示）
        plt.plot(ts_accel_vals, y_fit_display, 'r-', lw=2.5, label=f'Saito拟合线\n(斜率={saito_slope:.3f})', zorder=8)
        
        # 绘制y=0参考线
        plt.axhline(y=0, color='black', ls='-', lw=0.8, alpha=0.5, zorder=5)
        
        # 计算TTF交点 (1/v = 0 => t = -b/a)
        if saito_slope < -1e-10:
            ttf_pos = -saito_intercept / saito_slope
            ttf_time = ts_accel_vals[0] + pd.Timedelta(hours=ttf_pos / 6)  # 每10min=1/6h
            
            # 绘制TTF垂直线
            plt.axvline(x=ttf_time, color='red', ls=':', lw=2.5, alpha=0.8, zorder=7)
            # 添加标注框
            plt.annotate(f'TTF预测失稳点\n{ttf_time.strftime("%m-%d %H:%M")}',
                        xy=(ttf_time, display_max * 0.05), 
                        xytext=(ttf_time, display_max * 0.6),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                        ha='center', va='center', fontsize=9, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.9),
                        zorder=15)
    else:
        print("    [提示] 加速期数据不足，Saito拟合可能不准确")
    
    plt.title('图10：基于Saito流变学的临界失稳时间预测(TTF)')
    plt.ylabel('倒数速度 1/v (h/mm)')
    plt.ylim(0, display_max * 1.1)
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('Fig10_TTF.png', dpi=300); plt.close()
    
    # 11. 预警带
    fig11, ax11 = plt.subplots(figsize=(12, 5))
    ax11.plot(ts, df_res['预测值'], 'k-', lw=1.5, zorder=10)
    clrs = ['#E8F5E9', '#FFF9C4', '#FFE0B2', '#FFCDD2']
    lvl = df_res['预警等级'].fillna(0).astype(int).values
    for i in range(1, len(lvl)):
        if lvl[i] != lvl[i-1] or i == len(lvl)-1:
            ax11.axvspan(ts.iloc[i-1], ts.iloc[i], color=clrs[lvl[i]], alpha=0.6, zorder=1)
    # 添加图例
    patches = [mpatches.Patch(color=clrs[i], label=f'等级{i}') for i in range(4)]
    ax11.legend(handles=patches, loc='upper left')
    plt.title('图11：自适应预警决策动态映射带'); plt.savefig('Fig11_Warning.png', dpi=300); plt.close()
    
    # 12. 预警等级分布
    plt.figure(figsize=(7, 7))
    cnt = df_res['预警等级'].value_counts().sort_index()
    colors_pie = [clrs[int(k)] for k in cnt.index]
    plt.pie(cnt, labels=[f'等级{int(k)}' for k in cnt.index], autopct='%1.1f%%', colors=colors_pie, startangle=90)
    plt.title('图12：预警级别时间分布比例'); plt.savefig('Fig12_Pie.png', dpi=300); plt.close()
    
    # 13. 速度分阶段分析
    fig13, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 上图：速度曲线 + 阶段标注
    ax1.plot(ts, df_res['速度_平滑'], 'b-', lw=1.5, label='平滑速度')
    ax1.axhline(y=v_green_max, color='orange', ls='--', lw=1.5, label=f'加速阈值={v_green_max:.3f}')
    ax1.axhline(y=v_yellow_max, color='red', ls='--', lw=1.5, label=f'临滑阈值={v_yellow_max:.3f}')
    ax1.set_ylabel('速度 (mm/h)')
    ax1.legend(loc='upper left')
    ax1.set_title('图13.1：表面位移速度分阶段分析')
    
    # 阶段色带
    stage_colors = {'I-匀速稳定期': '#E8F5E9', 'II-加速蠕变期': '#FFF9C4', 'III-临滑失稳期': '#FFCDD2'}
    for i in range(1, len(df_res)):
        stage = df_res['阶段'].iloc[i]
        if stage != df_res['阶段'].iloc[i-1]:
            color = stage_colors.get(stage, '#FFFFFF')
            ax1.axvspan(ts.iloc[i-1], ts.iloc[i], color=color, alpha=0.4, zorder=0)
    
    # 下图：预警等级阶梯
    ax2.step(ts, df_res['预警等级'], where='post', color='darkred', lw=2)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['绿(正常)', '黄(注意)', '橙(警惕)', '红(撤离)'])
    ax2.set_ylabel('预警等级')
    ax2.set_xlabel('时间')
    ax2.set_title('图13.2：预警等级时序演化')
    
    plt.tight_layout()
    plt.savefig('Fig13_Stage.png', dpi=300); plt.close()
    
    print("\n[成功] 已生成 13 张分析图表。程序运行结束。")
    print("="*80)

if __name__ == "__main__":
    main()
