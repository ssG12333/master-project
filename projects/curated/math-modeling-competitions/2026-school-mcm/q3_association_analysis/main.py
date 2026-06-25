import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')

from q3_core_models import Q3DataProcessor, ImputerAndDetector, ShapEvaluator
from q3_visualizations import Q3Visualizations

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

OUTPUT_DIR = './q3_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
viz = Q3Visualizations(output_dir=OUTPUT_DIR)

def load_real_data_from_excel(excel_path):
    print(f">> 正在加载真实监测数据 (Excel)...\n文件路径: {excel_path}")
    print("   -> 正在读取 '训练集' Sheet...")
    df_train = pd.read_excel(excel_path, sheet_name='训练集')
    df_train.rename(columns={
        'a:降雨量_mm': 'Rainfall',
        'b:孔隙水压力_kPa': 'WaterPressure',
        'c:微震事件数': 'Seismic',
        'd:深部位移_mm': 'DeepDisp',
        'e:表面位移_mm': 'SurfaceDisp'
    }, inplace=True)
    df_train['Time'] = pd.date_range('2024-01-01', periods=len(df_train), freq='1H')
    train_dict = {
        'Rainfall': df_train[['Time', 'Rainfall']],
        'WaterPressure': df_train[['Time', 'WaterPressure']],
        'Seismic': df_train[['Time', 'Seismic']],
        'DeepDisp': df_train[['Time', 'DeepDisp']],
        'SurfaceDisp': df_train[['Time', 'SurfaceDisp']]
    }
    print("   -> 正在读取 '实验集' Sheet...")
    df_test = pd.read_excel(excel_path, sheet_name='实验集')
    df_test.rename(columns={
        '降雨量_mm': 'Rainfall',
        '孔隙水压力_kPa': 'WaterPressure',
        '微震事件数': 'Seismic',
        '深部位移_mm': 'DeepDisp',
        '表面位移_mm': 'SurfaceDisp'
    }, inplace=True)
    df_test['Time'] = pd.date_range(df_train['Time'].iloc[-1] + pd.Timedelta(hours=1), periods=len(df_test), freq='1H')
    return train_dict, df_test, df_train

def main():
    print("="*70)
    print("[边坡预警系统] 问题3：多维时序关联分析与贡献度量化")
    print("="*70)
    excel_path = "附件3：监测数据（训练集与实验集）-问题3.xlsx"
    try:
        raw_train_dict, raw_test_df, df_train_original = load_real_data_from_excel(excel_path)
    except Exception as e:
        print(f"[错误] 数据读取失败: {e}。\n提示: 请检查文件路径，并确保已安装 openpyxl (pip install openpyxl)。")
        return

    processor = Q3DataProcessor(target_freq='1H')
    df_aligned = processor.align_and_merge(raw_train_dict)
    missing_count = df_aligned.isnull().sum().sum()
    print(f"-> 原始训练集对齐后总缺失值数量: {missing_count}")

    # ==========================================
    # 步骤1: 探索性数据分析 (EDA)
    # ==========================================
    print("\n" + "="*70)
    print("[步骤1] 探索性数据分析 (EDA)")
    print("="*70)
    df_eda = df_aligned.copy()
    df_eda = df_eda.ffill().bfill()
    viz.plot_eda_timeseries(df_eda)
    viz.plot_eda_missing_heatmap(df_aligned)
    viz.plot_eda_correlation_matrix(df_eda)
    viz.plot_eda_variable_distributions(df_eda)
    viz.plot_eda_rainfall_deepdisp_lag(df_eda)
    viz.plot_eda_rainfall_surface_lag(df_eda)
    viz.plot_eda_waterpressure_surface_scatter(df_eda)
    print("[步骤1] EDA 图表生成完毕 (7张)")

    # ==========================================
    # 步骤2: 数据预处理 (插补 + 异常检测)
    # ==========================================
    print("\n" + "="*70)
    print("[步骤2] 数据预处理")
    print("="*70)
    imputer_detector = ImputerAndDetector(contamination=0.03)
    df_imputed = imputer_detector.impute_missing_values(df_aligned)
    viz.plot_preprocess_imputation_comparison(df_aligned, df_imputed)
    # 3.2异常检测：全部5个变量（含SurfaceDisp）
    all_monitor_features = ['Rainfall', 'WaterPressure', 'Seismic', 'DeepDisp', 'SurfaceDisp']
    anomaly_matrix = imputer_detector.detect_anomalies(df_imputed, all_monitor_features)
    viz.plot_preprocess_anomaly_detection(df_imputed, anomaly_matrix, all_monitor_features)
    df_features = processor.create_lag_features(df_imputed)
    
    # 3.3建模用的自变量（4个，不含SurfaceDisp因变量）
    model_features = ['Rainfall', 'WaterPressure', 'Seismic', 'DeepDisp']

    print("\n" + "="*70)
    print("[表3.1 填报用] 各监测变量单变量异常频次：")
    anomaly_counts = {}
    for feat in all_monitor_features:
        count = anomaly_matrix[f'{feat}_Anomaly'].sum()
        anomaly_counts[feat] = count
        print(f"   - {feat:15}: 识别出 {count} 个异常点")
    print("-" * 70)
    joint_anomalies = anomaly_matrix[anomaly_matrix['Is_Joint_Anomaly'] == 1]
    print(f"[表3.2 填报用] 提取到的共同异常时间点 (共 {len(joint_anomalies)} 个，截取前5个展示)：")
    for t in joint_anomalies.index[:5]:
        involved_vars = [feat for feat in all_monitor_features if anomaly_matrix.loc[t, f'{feat}_Anomaly'] == 1]
        print(f"   - 时间: {t}, 涉及异常变量: {', '.join(involved_vars)}")
    print("="*70)
    
    # 生成表3.1和表3.2并保存为Excel文件
    print("\n" + "="*70)
    print("[正在生成表3.1和表3.2 Excel文件...]")
    print("="*70)
    
    # 变量字母映射
    var_letter_map = {
        'Rainfall': 'a',
        'WaterPressure': 'b',
        'Seismic': 'c',
        'DeepDisp': 'd',
        'SurfaceDisp': 'e'
    }
    
    # 构建表3.1
    table_3_1_data = []
    total_anomalies = 0
    for feat in all_monitor_features:
        letter = var_letter_map.get(feat, '')
        count = anomaly_counts[feat]
        total_anomalies += count
        table_3_1_data.append({
            '数据集变量': f'{letter}：{feat}',
            '异常点数量': count
        })
    table_3_1_data.append({
        '数据集变量': '总数',
        '异常点数量': total_anomalies
    })
    df_table_3_1 = pd.DataFrame(table_3_1_data)
    
    # 构建表3.2
    table_3_2_data = []
    for idx, t in enumerate(joint_anomalies.index, 1):
        involved_vars = [feat for feat in all_monitor_features if anomaly_matrix.loc[t, f'{feat}_Anomaly'] == 1]
        var_letters = ''.join([var_letter_map.get(feat, '') for feat in involved_vars])
        table_3_2_data.append({
            '时间点对应编号': idx,
            '共同异常点处的异常变量': var_letters
        })
    df_table_3_2 = pd.DataFrame(table_3_2_data)
    
    # 保存为Excel文件（包含两个Sheet）
    table_output_path = os.path.join(OUTPUT_DIR, "表3.1和表3.2_异常检测结果.xlsx")
    with pd.ExcelWriter(table_output_path, engine='openpyxl') as writer:
        df_table_3_1.to_excel(writer, sheet_name='表3.1_单变量异常点', index=False)
        df_table_3_2.to_excel(writer, sheet_name='表3.2_共同异常点', index=False)
    
    print(f"[完成] 表3.1和表3.2已保存至: {table_output_path}")
    print(f"   - 表3.1: 共{len(all_monitor_features)}个变量，总异常点{total_anomalies}个")
    print(f"   - 表3.2: 共{len(joint_anomalies)}个共同异常点")
    print("="*70)
    
    print("[步骤2] 预处理图表生成完毕 (2张)")

    # ==========================================
    # 步骤3: 模型训练与验证 (6-4 时间划分)
    # ==========================================
    print("\n" + "="*70)
    print("[步骤3] 模型训练与验证 (6-4 时间划分)")
    print("="*70)
    # 3.3建模特征：4个自变量 + 滞后特征（不含SurfaceDisp因变量）
    train_features = model_features + ['Rainfall_sum_24h', 'Rainfall_lag_12h', 'WaterPressure_lag_6h', 'Seismic_lag_2h']
    df_clean = df_features.dropna(subset=train_features + ['SurfaceDisp'])
    n_total = len(df_clean)
    n_train = int(n_total * 0.6)
    train_idx = range(n_train)
    val_idx = range(n_train, n_total)
    df_train_model = df_clean.iloc[:n_train]
    df_val_model = df_clean.iloc[n_train:]
    X_train, y_train = df_train_model[train_features], df_train_model['SurfaceDisp']
    X_val, y_val = df_val_model[train_features], df_val_model['SurfaceDisp']
    viz.plot_validation_data_split(train_idx, val_idx, df_clean.index)
    evaluator = ShapEvaluator(target_col='SurfaceDisp')
    evaluator.model.fit(X_train, y_train)
    train_r2 = evaluator.model.score(X_train, y_train)
    print(f"-> 训练集 R2: {train_r2:.4f}")
    y_val_pred = evaluator.model.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    print(f"-> 验证集 R2: {val_r2:.4f}")
    print(f"-> 验证集 RMSE: {val_rmse:.2f}")
    print(f"-> 验证集 MAE: {val_mae:.2f}")
    viz.plot_validation_pred_vs_actual_scatter(y_val, y_val_pred)
    viz.plot_validation_timeseries_comparison(df_val_model.index, y_val, y_val_pred)
    viz.plot_validation_residual_histogram(y_val, y_val_pred)
    viz.plot_validation_residual_vs_pred(y_val_pred, y_val)
    print("[步骤3] 验证图表生成完毕 (5张)")

    # ==========================================
    # 步骤4: 全量训练 + SHAP 归因分析
    # ==========================================
    print("\n" + "="*70)
    print("[步骤4] SHAP 归因分析 (全量训练)")
    print("="*70)
    evaluator_full = ShapEvaluator(target_col='SurfaceDisp')
    X_full, y_full = evaluator_full.train_and_evaluate(df_features, train_features)
    shap_values, explainer, pct_dict = evaluator_full.calculate_shap(X_full)
    print("\n" + "="*70)
    print("[贡献度分析] 各诱发因素对表面位移的总体贡献度 (SHAP 值归一化)：")
    sorted_pct = sorted(pct_dict.items(), key=lambda item: item[1], reverse=True)
    for feat, pct in sorted_pct:
        print(f"   >>> {feat:20}: {pct:.2f} %")
    print("="*70)
    viz.plot_shap_beeswarm(shap_values, X_full)
    viz.plot_shap_feature_importance_bar(pct_dict)
    viz.plot_shap_dependence_top3(shap_values, X_full, pct_dict)
    print("[步骤4] SHAP 图表生成完毕 (3张)")

    # ==========================================
    # 步骤5: 实验集预测 + 综合对比图
    # ==========================================
    print("\n" + "="*70)
    print("[步骤5] 实验集预测")
    print("="*70)
    test_dict = {
        'Rainfall': raw_test_df[['Time', 'Rainfall']],
        'WaterPressure': raw_test_df[['Time', 'WaterPressure']],
        'Seismic': raw_test_df[['Time', 'Seismic']],
        'DeepDisp': raw_test_df[['Time', 'DeepDisp']],
        'SurfaceDisp': raw_test_df[['Time', 'SurfaceDisp']]
    }
    df_test_aligned = processor.align_and_merge(test_dict)
    df_test_imputed = imputer_detector.impute_missing_values(df_test_aligned, target_col='SurfaceDisp')
    df_test_features = processor.create_lag_features(df_test_imputed)
    X_test = df_test_features[train_features].bfill().ffill()
    final_predictions = evaluator_full.model.predict(X_test)

    # 训练集预测 (用于对比图)
    train_pred = evaluator_full.model.predict(X_full)

    viz.plot_train_test_prediction_comparison(
        train_time=df_clean.iloc[:n_train].index,
        train_actual=y_train,
        train_pred=evaluator.model.predict(X_train),
        val_time=df_clean.iloc[n_train:].index,
        val_actual=y_val,
        val_pred=y_val_pred,
        test_time=raw_test_df['Time'].values,
        test_pred=final_predictions
    )
    viz.plot_test_prediction_boxplot(final_predictions)
    viz.plot_test_prediction_scatter(raw_test_df['Time'].values, final_predictions)

    raw_test_df['SurfaceDisp_Predicted'] = final_predictions
    raw_test_df_out = raw_test_df.drop(columns=['Time'])
    output_filename = os.path.join(OUTPUT_DIR, "问题3_实验集_表面位移预测结果.csv")
    raw_test_df_out.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"[完成] 实验集表面位移预测已完成！结果已导出至: {output_filename}")
    print("[步骤5] 预测图表生成完毕 (2张)")

    print("\n" + "="*70)
    print("[完成] 问题三全部流水线执行完毕！")
    print(f"所有图表已保存至目录: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
