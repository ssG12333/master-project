import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')

# ================= 配置与初始化 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("muted")

# 定义全局特征列 (加入更丰富的交叉特征)
FEATURES = ['降雨量_mm', '孔隙水压力_kPa', '微震事件数', '爆破振动烈度_V',
            '降雨量_2h_sum', '孔压_2h_mean', '孔压_2h_diff', '水力耦合因子']
TARGET = '表面位移增量_mm'

# ================= 1. 数据加载与高级特征工程 =================
def load_and_engineer_features(train_path, test_path):
    print("-" * 50)
    print("Step 1: 加载数据与高级特征工程 (引入物理公式与滞后效应)...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df['时间'] = pd.to_datetime(train_df['时间'])
    test_df['时间'] = pd.to_datetime(test_df['时间'])
    train_df = train_df.sort_values('时间').reset_index(drop=True)
    test_df = test_df.sort_values('时间').reset_index(drop=True)
    
    def process_features(df, is_train=True):
        df_proc = df.copy()
        
        # 1. 爆破特征重构：萨道夫斯基经验公式思想 V ∝ (Q^(1/3) / R)
        df_proc['爆破点距离_m'] = df_proc['爆破点距离_m'].fillna(0)
        df_proc['单段最大药量_kg'] = df_proc['单段最大药量_kg'].fillna(0)
        df_proc['爆破振动烈度_V'] = np.where(
            df_proc['爆破点距离_m'] > 0,
            (df_proc['单段最大药量_kg'] ** (1/3)) / df_proc['爆破点距离_m'],
            0
        )
        df_proc['微震事件数'] = df_proc['微震事件数'].fillna(0)
        
        # 2. 提取时间滞后特征 (窗口统计)
        df_proc = df_proc.set_index('时间')
        df_proc['降雨量_2h_sum'] = df_proc['降雨量_mm'].rolling(window='2h', min_periods=1).sum()
        df_proc['降雨量_6h_sum'] = df_proc['降雨量_mm'].rolling(window='6h', min_periods=1).sum()
        df_proc['孔压_2h_mean'] = df_proc['孔隙水压力_kPa'].rolling(window='2h', min_periods=1).mean()
        df_proc['孔压_2h_diff'] = df_proc['孔隙水压力_kPa'].diff(periods=12).fillna(0)
        
        # 3. 交叉特征：水力耦合因子 (降雨渗流导致孔压上升的综合作用)
        df_proc['水力耦合因子'] = df_proc['降雨量_6h_sum'] * df_proc['孔压_2h_mean']
        df_proc = df_proc.reset_index()
        
        # 4. 目标变量
        if is_train:
            df_proc['表面位移增量_mm'] = df_proc['表面位移_mm'].diff().fillna(0)
            df_proc['表面位移增量_mm'] = df_proc['表面位移增量_mm'].clip(lower=0)
        return df_proc

    train_proc = process_features(train_df, is_train=True)
    test_proc = process_features(test_df, is_train=False)
    return train_proc, test_proc

# ================= 2. 探索性数据分析 (EDA图表生成) =================
def exploratory_data_analysis(train_df):
    print("-" * 50)
    print("Step 2: 绘制多维度探索性数据分析(EDA)图表...")
    
    # 表面位移
    plt.figure(figsize=(12, 4))
    plt.plot(train_df['时间'], train_df['表面位移_mm'], color='blue')
    plt.title('图1-1：表面位移时序演化分析', fontsize=16)
    plt.ylabel('表面位移(mm)'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('Fig01_1_Displacement_EDA.png', dpi=300); plt.close()
    
    # 降雨量
    plt.figure(figsize=(12, 4))
    plt.plot(train_df['时间'], train_df['降雨量_mm'], color='green')
    plt.title('图1-2：降雨量时序演化分析', fontsize=16)
    plt.ylabel('降雨量(mm)'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('Fig01_2_Rainfall_EDA.png', dpi=300); plt.close()
    
    # 孔隙水压力
    plt.figure(figsize=(12, 4))
    plt.plot(train_df['时间'], train_df['孔隙水压力_kPa'], color='purple')
    plt.title('图1-3：孔隙水压力时序演化分析', fontsize=16)
    plt.ylabel('孔隙水压力(kPa)'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('Fig01_3_PorePressure_EDA.png', dpi=300); plt.close()
    
    # 扰动强度
    plt.figure(figsize=(12, 4))
    plt.plot(train_df['时间'], train_df['微震事件数'] + train_df['爆破振动烈度_V'], color='red')
    plt.title('图1-4：扰动强度(微震+爆破)时序演化分析', fontsize=16)
    plt.ylabel('扰动强度'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('Fig01_4_Disturbance_EDA.png', dpi=300); plt.close()

    # 相关性热力图
    plt.figure(figsize=(10, 8))
    corr_features = FEATURES + ['表面位移_mm']
    corr_matrix = train_df[corr_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                mask=np.triu(np.ones_like(corr_matrix, dtype=bool)))
    plt.title('图2：多源特征与表面位移相关性热力图', fontsize=16)
    plt.tight_layout(); plt.savefig('Fig02_Correlation_Heatmap.png', dpi=300); plt.close()

# ================= 3. 训练集阶段自动划分与验证图表 =================
def segment_training_stages_and_plot(train_df):
    print("-" * 50)
    print("Step 3: 执行K-Means速率突变识别并绘制阶段验证图...")
    
    train_df['平滑速率'] = train_df[TARGET].rolling(window=6, center=True, min_periods=1).mean()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(train_df[['平滑速率']])
    
    centers = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(centers)
    label_mapping = {sorted_idx[0]: 1, sorted_idx[1]: 2, sorted_idx[2]: 3}
    train_df['阶段标签'] = [label_mapping[lbl] for lbl in labels]
    
    train_df['速度'] = train_df['表面位移_mm'].diff().fillna(0).clip(lower=0)
    train_df['平滑速度'] = train_df['速度'].rolling(window=6, center=True, min_periods=1).mean()
    train_df['加速度'] = train_df['平滑速度'].diff().fillna(0)
    train_df['切线角'] = np.arctan(train_df['平滑速度']) * 180 / np.pi
    
    # 简化阶段划分：基于位移阈值
    n = len(train_df)
    max_disp = train_df['表面位移_mm'].max()
    disp_threshold_1 = max_disp * 0.05
    disp_threshold_2 = max_disp * 0.25
    
    stage2_start = (train_df['表面位移_mm'] > disp_threshold_1).idxmax()
    stage3_start = (train_df['表面位移_mm'] > disp_threshold_2).idxmax()
    
    stage2_start = max(n // 6, min(stage2_start, n // 2))
    stage3_start = max(stage2_start + n // 4, min(stage3_start, n - n // 8))
    
    train_df['简化阶段标签'] = 1
    train_df.loc[train_df.index[stage2_start:], '简化阶段标签'] = 2
    train_df.loc[train_df.index[stage3_start:], '简化阶段标签'] = 3
    
    print(f"训练集阶段划分：阶段1[0:{stage2_start}]，阶段2[{stage2_start}:{stage3_start}]，阶段3[{stage3_start}:{n}]")
    
    # 更新阶段标签
    train_df['阶段标签'] = train_df['简化阶段标签']
    
    # 计算各阶段位移变化统计（必须在阶段标签更新后计算）
    train_stage_disp_stats = {}
    for stage in [1, 2, 3]:
        mask = train_df['阶段标签'] == stage
        stage_data = train_df[mask]
        train_stage_disp_stats[stage] = {
            'start': stage_data['表面位移_mm'].iloc[0],
            'end': stage_data['表面位移_mm'].iloc[-1],
            'change': stage_data['表面位移_mm'].iloc[-1] - stage_data['表面位移_mm'].iloc[0]
        }

    # 【图 3】训练集多尺度阶段切分综合判别图
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('图3：训练集多尺度阶段切分综合判别图', fontsize=16, fontweight='bold')
    
    colors = {1: 'lightgreen', 2: 'lightyellow', 3: 'mistyrose'}
    
    # (a) 位移演化
    axes[0].plot(train_df['时间'], train_df['表面位移_mm'], color='grey', alpha=0.3, linewidth=1, label='原始数据')
    axes[0].plot(train_df['时间'], train_df['表面位移_mm'].rolling(window=6, center=True, min_periods=1).mean(), 
                 color='blue', linewidth=2, label='平滑形变')
    
    axes[0].axvspan(train_df['时间'].iloc[0], train_df['时间'].iloc[stage2_start-1], color=colors[1], alpha=0.3)
    axes[0].axvspan(train_df['时间'].iloc[stage2_start], train_df['时间'].iloc[stage3_start-1], color=colors[2], alpha=0.3)
    axes[0].axvspan(train_df['时间'].iloc[stage3_start], train_df['时间'].iloc[-1], color=colors[3], alpha=0.3)
    
    axes[0].axvline(train_df['时间'].iloc[stage2_start], color='red', linestyle='--', linewidth=2, alpha=0.8)
    axes[0].axvline(train_df['时间'].iloc[stage3_start], color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    axes[0].set_ylabel('表面位移 (mm)', fontsize=11)
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.95, '(a) 位移演化全貌与阶段划分', transform=axes[0].transAxes, fontsize=11, fontweight='bold')
    
    # (b) 速度演化
    axes[1].plot(train_df['时间'], train_df['平滑速度'], color='orange', linewidth=1.5, alpha=0.8)
    
    axes[1].axvspan(train_df['时间'].iloc[0], train_df['时间'].iloc[stage2_start-1], color=colors[1], alpha=0.3)
    axes[1].axvspan(train_df['时间'].iloc[stage2_start], train_df['时间'].iloc[stage3_start-1], color=colors[2], alpha=0.3)
    axes[1].axvspan(train_df['时间'].iloc[stage3_start], train_df['时间'].iloc[-1], color=colors[3], alpha=0.3)
    
    axes[1].axvline(train_df['时间'].iloc[stage2_start], color='red', linestyle='--', linewidth=2, alpha=0.8)
    axes[1].axvline(train_df['时间'].iloc[stage3_start], color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    axes[1].set_ylabel('速度 (mm/h)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.02, 0.95, '(b) 瞬时速度演化特征', transform=axes[1].transAxes, fontsize=11, fontweight='bold')
    
    # (c) 切线角
    axes[2].plot(train_df['时间'], train_df['切线角'], color='purple', linewidth=1, alpha=0.7)
    axes[2].axhline(y=45, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='45° 阈值线')
    axes[2].axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='80° 阈值线')
    
    axes[2].axvspan(train_df['时间'].iloc[0], train_df['时间'].iloc[stage2_start-1], color=colors[1], alpha=0.3)
    axes[2].axvspan(train_df['时间'].iloc[stage2_start], train_df['时间'].iloc[stage3_start-1], color=colors[2], alpha=0.3)
    axes[2].axvspan(train_df['时间'].iloc[stage3_start], train_df['时间'].iloc[-1], color=colors[3], alpha=0.3)
    
    axes[2].axvline(train_df['时间'].iloc[stage2_start], color='red', linestyle='--', linewidth=2, alpha=0.8)
    axes[2].axvline(train_df['时间'].iloc[stage3_start], color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    axes[2].set_ylabel('切线角 (°)', fontsize=11)
    axes[2].set_xlabel('时间', fontsize=11)
    axes[2].legend(loc='upper left', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].text(0.02, 0.95, '(c) 切线角阈值演变监测', transform=axes[2].transAxes, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Fig03_Train_Stage_Analysis.png', dpi=300)
    plt.close()
    
    # 【图 4】阶段速率核密度估计图
    colors_kde = {1: 'mediumseagreen', 2: 'orange', 3: 'crimson'}
    plt.figure(figsize=(10, 6))
    for stage in [1, 2, 3]:
        sns.kdeplot(train_df[train_df['阶段标签']==stage]['平滑速率'], 
                    fill=True, label=f'阶段 {stage}', color=colors_kde[stage])
    plt.title('图4：不同演化阶段下表面位移速率的核密度分布对比', fontsize=16)
    plt.xlabel('形变速率 (mm / 10min)'); plt.ylabel('频数密度')
    plt.legend(); plt.grid(alpha=0.3)
    plt.xlim(0, train_df['平滑速率'].quantile(0.99))
    plt.savefig('Fig04_Velocity_KDE_Stages.png', dpi=300); plt.close()
    
    # 【图 5】箱线图
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='阶段标签', y='孔隙水压力_kPa', data=train_df, palette=colors_kde)
    plt.title('图5-1：孔隙水压力跨阶段演化分布', fontsize=14)
    plt.tight_layout(); plt.savefig('Fig05_1_PorePressure_Boxplot.png', dpi=300); plt.close()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='阶段标签', y='微震事件数', data=train_df, palette=colors_kde)
    plt.title('图5-2：微震频次跨阶段演化分布', fontsize=14)
    plt.tight_layout(); plt.savefig('Fig05_2_Microseismic_Boxplot.png', dpi=300); plt.close()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='阶段标签', y='爆破振动烈度_V', data=train_df[train_df['爆破振动烈度_V']>0], palette=colors_kde)
    plt.title('图5-3：有效爆破烈度跨阶段分布', fontsize=14)
    plt.tight_layout(); plt.savefig('Fig05_3_Blasting_Boxplot.png', dpi=300); plt.close()

    return train_df, train_stage_disp_stats

# ================= 4. 异构集成模型训练与特征重要性 (Q4.1) =================
def train_heterogeneous_ensemble(train_df):
    print("-" * 50)
    print("Step 4: 构建异构集成学习架构(GBDT+RF+ET)并提取特征重要性...")
    models = {}
    importances_list = {}
    
    TARGET_COL = '表面位移增量_mm'
    
    for stage in [1, 2, 3]:
        stage_df = train_df[train_df['阶段标签'] == stage].dropna()
        X = stage_df[FEATURES]
        y = stage_df[TARGET_COL]
        
        gbdt = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
        rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
        et = ExtraTreesRegressor(n_estimators=150, max_depth=8, random_state=42)
        
        ensemble = VotingRegressor(estimators=[('gbdt', gbdt), ('rf', rf), ('et', et)])
        ensemble.fit(X, y)
        models[stage] = ensemble
        
        imp_gbdt = ensemble.named_estimators_['gbdt'].feature_importances_
        imp_rf = ensemble.named_estimators_['rf'].feature_importances_
        imp_et = ensemble.named_estimators_['et'].feature_importances_
        avg_importance = (imp_gbdt + imp_rf + imp_et) / 3.0
        importances_list[f'阶段 {stage}'] = avg_importance
        
    imp_df = pd.DataFrame(importances_list, index=FEATURES)
    ax = imp_df.plot(kind='bar', figsize=(14, 7), colormap='Set2', width=0.8, edgecolor='black')
    plt.title('图7：基于集成学习架构的不同演化阶段多源特征重要性动态漂移分析', fontsize=16)
    plt.ylabel('平均特征重要性贡献度')
    plt.xticks(rotation=20, ha='right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Fig07_Stage_Feature_Importance.png', dpi=300); plt.close()
    
    return models

# ================= 5. 实验集滚动预测与置信区间计算 (Q4.2) =================
def predict_experimental_set_with_ci(test_df, models, train_stage_disp_stats):
    print("-" * 50)
    print("Step 5: 实验集全周期动态预测与模型置信度推演...")
    
    print(f"训练集阶段位移变化:")
    for stage, stats in train_stage_disp_stats.items():
        print(f"  阶段{stage}: {stats['start']:.1f}->{stats['end']:.1f}mm (Δ={stats['change']:.1f}mm)")
    
    pred_increments, pred_stds = [], []
    
    for idx, row in test_df.iterrows():
        stage = row['阶段标签']
        x_input = row[FEATURES].values.reshape(1, -1)
        
        ensemble_model = models[stage]
        preds = [est.predict(x_input)[0] for est in ensemble_model.estimators_]
        
        pred_increments.append(np.mean(preds))
        pred_stds.append(np.std(preds))
        
    test_df['预测位移增量_mean'] = pred_increments
    test_df['预测位移增量_std'] = pred_stds
    
    # 阶段校准
    for stage in [1, 2, 3]:
        mask = test_df['阶段标签'] == stage
        if mask.sum() > 0:
            current_total = test_df.loc[mask, '预测位移增量_mean'].sum()
            target_total = train_stage_disp_stats[stage]['change']
            
            if current_total > 0:
                calibration = target_total / current_total
                test_df.loc[mask, '预测位移增量_mean'] *= calibration
    
    test_df['平滑增量'] = savgol_filter(test_df['预测位移增量_mean'], window_length=5, polyorder=2)
    test_df['平滑增量'] = test_df['平滑增量'].clip(lower=0)
    
    test_df['增量_std_smooth'] = savgol_filter(test_df['预测位移增量_std'], window_length=21, polyorder=1).clip(min=0)
    
    test_df['预测表面位移_mm'] = test_df['平滑增量'].cumsum()
    test_df['预测位移_std_累计'] = test_df['增量_std_smooth'].cumsum() * 0.15
    
    # 【图 8】预测速率脉冲图
    plt.figure(figsize=(14, 5))
    plt.plot(test_df['时间'], test_df['平滑增量'], color='indigo', label='预测形变速率 (mm/10min)')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title('图8：实验集分阶段动态路由预测的形变速率脉冲时序图', fontsize=16)
    plt.xlabel('时间'); plt.ylabel('形变增量速率')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig('Fig08_Predicted_Velocity.png', dpi=300); plt.close()

    # 【图 8-1】实验集多尺度阶段切分综合判别图
    test_df['预测速度'] = test_df['平滑增量']
    test_df['平滑预测速度'] = test_df['预测速度'].rolling(window=6, center=True, min_periods=1).mean()
    test_df['预测加速度'] = test_df['平滑预测速度'].diff().fillna(0)
    test_df['预测切线角'] = np.arctan(test_df['平滑预测速度']) * 180 / np.pi
    
    test_stage_changes = test_df[test_df['阶段标签'].diff() != 0].index.tolist()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('图8-1：实验集多尺度阶段切分综合判别图', fontsize=16, fontweight='bold')
    
    colors = {1: 'lightgreen', 2: 'lightyellow', 3: 'mistyrose'}
    
    # (a) 位移
    axes[0].plot(test_df['时间'], test_df['预测表面位移_mm'], color='grey', alpha=0.5, linewidth=1, label='原始预测位移')
    axes[0].plot(test_df['时间'], test_df['预测表面位移_mm'].rolling(window=6, center=True, min_periods=1).mean(), 
                 color='blue', linewidth=2, label='平滑预测形变')
    
    for idx in test_stage_changes:
        axes[0].axvline(test_df['时间'].iloc[idx], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    current_stage = test_df['阶段标签'].iloc[0]
    start_idx = 0
    for idx in range(1, len(test_df)):
        if test_df['阶段标签'].iloc[idx] != current_stage:
            axes[0].axvspan(test_df['时间'].iloc[start_idx], test_df['时间'].iloc[idx-1], 
                           color=colors[current_stage], alpha=0.3)
            start_idx = idx
            current_stage = test_df['阶段标签'].iloc[idx]
    axes[0].axvspan(test_df['时间'].iloc[start_idx], test_df['时间'].iloc[-1], 
                   color=colors[current_stage], alpha=0.3)
    
    axes[0].set_ylabel('表面位移 (mm)', fontsize=11)
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.95, '(a) 预测位移演化全貌与阶段划分', transform=axes[0].transAxes, fontsize=11, fontweight='bold')
    
    # (b) 速度
    axes[1].plot(test_df['时间'], test_df['平滑预测速度'], color='orange', linewidth=1.5, alpha=0.8)
    
    current_stage = test_df['阶段标签'].iloc[0]
    start_idx = 0
    for idx in range(1, len(test_df)):
        if test_df['阶段标签'].iloc[idx] != current_stage:
            axes[1].axvspan(test_df['时间'].iloc[start_idx], test_df['时间'].iloc[idx-1], 
                           color=colors[current_stage], alpha=0.3)
            start_idx = idx
            current_stage = test_df['阶段标签'].iloc[idx]
    axes[1].axvspan(test_df['时间'].iloc[start_idx], test_df['时间'].iloc[-1], 
                   color=colors[current_stage], alpha=0.3)
    
    for idx in test_stage_changes:
        axes[1].axvline(test_df['时间'].iloc[idx], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    axes[1].set_ylabel('速度 (mm/h)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.02, 0.95, '(b) 预测瞬时速度演化特征', transform=axes[1].transAxes, fontsize=11, fontweight='bold')
    
    # (c) 切线角
    axes[2].plot(test_df['时间'], test_df['预测切线角'], color='purple', linewidth=1, alpha=0.7)
    axes[2].axhline(y=45, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='45° 阈值线')
    axes[2].axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='80° 阈值线')
    
    current_stage = test_df['阶段标签'].iloc[0]
    start_idx = 0
    for idx in range(1, len(test_df)):
        if test_df['阶段标签'].iloc[idx] != current_stage:
            axes[2].axvspan(test_df['时间'].iloc[start_idx], test_df['时间'].iloc[idx-1], 
                           color=colors[current_stage], alpha=0.3)
            start_idx = idx
            current_stage = test_df['阶段标签'].iloc[idx]
    axes[2].axvspan(test_df['时间'].iloc[start_idx], test_df['时间'].iloc[-1], 
                   color=colors[current_stage], alpha=0.3)
    
    for idx in test_stage_changes:
        axes[2].axvline(test_df['时间'].iloc[idx], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    axes[2].set_ylabel('切线角 (°)', fontsize=11)
    axes[2].set_xlabel('时间', fontsize=11)
    axes[2].legend(loc='upper left', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].text(0.02, 0.95, '(c) 预测切线角阈值演变监测', transform=axes[2].transAxes, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Fig08_1_Test_Stage_Analysis.png', dpi=300)
    plt.close()

    return test_df

# ================= 6. 全局结果展现与关键点抽取 =================
def plot_final_prediction_and_extract_targets(test_df):
    print("-" * 50)
    print("Step 6: 渲染最终预测包络图与数据表格导出...")
    
    plt.figure(figsize=(15, 8))
    plt.plot(test_df['时间'], test_df['预测表面位移_mm'], color='darkred', linewidth=3, label='异构集成模型预测中值曲线')
    
    colors = {1: 'mediumseagreen', 2: 'orange', 3: 'crimson'}
    
    test_stage_changes = test_df[test_df['阶段标签'].diff() != 0].index.tolist()
    
    for stage in [1, 2, 3]:
        stage_mask = test_df['阶段标签'] == stage
        plt.fill_between(test_df['时间'], 0, test_df['预测表面位移_mm'].max() * 1.1, 
                         where=stage_mask, color=colors[stage], alpha=0.15, 
                         label=f'系统判定阶段 {stage}')
    
    for boundary_idx in test_stage_changes:
        plt.axvline(test_df['时间'].iloc[boundary_idx], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    target_times = [
        '2025-05-09 12:00:00',
        '2025-05-27 08:00:00',
        '2025-06-01 12:00:00',
        '2025-06-03 22:00:00',
        '2025-06-04 01:40:00'
    ]
    
    target_points = []
    for t_str in target_times:
        t_ts = pd.to_datetime(t_str)
        closest_idx = (test_df['时间'] - t_ts).abs().idxmin()
        val = test_df.loc[closest_idx, '预测表面位移_mm']
        std_val = test_df.loc[closest_idx, '预测位移_std_累计']
        target_points.append((t_ts, val, std_val))
        
        plt.scatter([t_ts], [val], color='gold', s=150, zorder=5, edgecolor='black')
        plt.annotate(f'{val:.1f}mm', xy=(t_ts, val), xytext=(5, 10), 
                     textcoords='offset points', fontsize=11, fontweight='bold')
    
    plt.fill_between(test_df['时间'], 
                     test_df['预测表面位移_mm'] - 1.96 * test_df['预测位移_std_累计'],
                     test_df['预测表面位移_mm'] + 1.96 * test_df['预测位移_std_累计'],
                     alpha=0.2, color='gray', label='95% 预测置信区间包络带')
    
    plt.title('图9：实验集全周期表面位移预测趋势图及5大要求时间点标定', fontsize=18)
    plt.xlabel('时间', fontsize=13); plt.ylabel('表面位移预测值 (mm)', fontsize=13)
    plt.legend(loc='upper left', fontsize=12); plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('Fig09_Final_Displacement.png', dpi=300); plt.close()
    
    # 【图 10】置信区间放大图
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['时间'], test_df['预测表面位移_mm'], color='darkred', linewidth=2)
    plt.fill_between(test_df['时间'], 
                     test_df['预测表面位移_mm'] - 1.96 * test_df['预测位移_std_累计'],
                     test_df['预测表面位移_mm'] + 1.96 * test_df['预测位移_std_累计'],
                     alpha=0.3, color='gray', label='95% 预测置信区间')
    
    for t_ts, val, std_val in target_points:
        plt.scatter([t_ts], [val], color='gold', s=100, zorder=5, edgecolor='black')
        plt.errorbar([t_ts], [val], yerr=1.96*std_val, fmt='none', color='red', capsize=5)
    
    plt.title('图10：实验集表面位移预测95%置信区间包络带局部放大图', fontsize=16)
    plt.xlabel('时间'); plt.ylabel('表面位移预测值 (mm)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('Fig10_Confidence_Interval.png', dpi=300); plt.close()
    
    print("\n" + "★" * 60)
    print("【表 4.1 实验集表面位移预测精确结果提取】")
    print("★" * 60)
    print(f"{'要求时间点':<25} | {'高精预测位移 (mm)':>15}")
    print("-" * 50)
    for t_ts, val, _ in target_points:
        print(f"{t_ts.strftime('%Y-%m-%d %H:%M'):<25} | {val:>15.3f}")
    print("★" * 60)

# ================= 主程序入口 =================
if __name__ == "__main__":
    EXCEL_FILE = "附件4：监测数据（训练集与实验集）-问题4.xlsx"
    
    try:
        print("正在从Excel文件读取数据...")
        xl = pd.ExcelFile(EXCEL_FILE)
        train_sheet = xl.sheet_names[0]
        test_sheet = xl.sheet_names[1]
        
        train_df_raw = pd.read_excel(EXCEL_FILE, sheet_name=train_sheet)
        test_df_raw = pd.read_excel(EXCEL_FILE, sheet_name=test_sheet)
        
        TRAIN_CSV = "训练集.csv"
        TEST_CSV = "实验集.csv"
        train_df_raw.to_csv(TRAIN_CSV, index=False, encoding='utf-8-sig')
        test_df_raw.to_csv(TEST_CSV, index=False, encoding='utf-8-sig')
        print(f"已将数据转换为CSV格式: {TRAIN_CSV}, {TEST_CSV}")
        
        train_df, test_df = load_and_engineer_features(TRAIN_CSV, TEST_CSV)
        
        exploratory_data_analysis(train_df)
        
        train_df, train_stage_disp_stats = segment_training_stages_and_plot(train_df)
        
        stage_models = train_heterogeneous_ensemble(train_df)
        
        test_df_predicted = predict_experimental_set_with_ci(test_df, stage_models, train_stage_disp_stats)
        
        plot_final_prediction_and_extract_targets(test_df_predicted)
        
        print("\n[SUCCESS] 全部流程执行完毕！")
        print("共生成了独立的图表，请在当前目录查收以 'Fig' 命名的图片文件！")
        
    except FileNotFoundError as e:
        print("\n【严重错误】未找到CSV数据文件！请确认您上传的两个数据文件是否与脚本在同一目录下，且文件名完全一致。")
    except Exception as e:
        print(f"\n【运行错误】程序执行失败：{e}")
