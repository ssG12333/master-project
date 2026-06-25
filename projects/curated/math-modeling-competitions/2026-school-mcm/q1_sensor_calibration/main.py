import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import time

from 数据处理 import DataProcessor
from 模型算法部分 import RidgeRegressor, PolynomialRegressor, SVRRegressor
from 出图 import Q1Visualizer

def main():
    start_time = time.time()
    os.makedirs('eda_plots', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    print("="*70)
    print("  边坡灾害智能预警系统 - 问题1 数据校正管线")
    print("="*70)
    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'C 附件(Attachment)')
    filepath = os.path.join(base_dir, '附件1：两组位移时序数据-问题1.xlsx')
    
    if not os.path.exists(filepath):
        alt_path = os.path.join(base_dir, 'Attachment 1：Two sets of displacement time series data – Question 1.xlsx')
        if os.path.exists(alt_path):
            filepath = alt_path
    
    processor = DataProcessor(filepath)
    try:
        df = processor.load_data()
    except FileNotFoundError:
        print(f"[错误] 找不到文件: {filepath}")
        return
    
    print("\n" + "="*70)
    print("  Step 1: 数据探索性分析 (EDA)")
    print("="*70)
    processor.plot_eda(save_dir="eda_plots")
    
    print("\n" + "="*70)
    print("  Step 2: CEEMDAN 去噪与特征提取")
    print("="*70)
    
    print(f"[全量] 使用全部 {len(df)} 个数据点")
    imfs, a_smooth = processor.apply_ceemdan(df['A'], max_imf=5)
    df_sample = processor.extract_features(df, a_smooth)
    print(f"[特征提取] 完成! 特征列: {[c for c in df_sample.columns if c not in ['Time', 'A', 'B', 'T_minutes']]}")
    
    print("\n" + "="*70)
    print("  Step 3: 多模型训练与对比")
    print("="*70)
    
    models = {}
    
    print("\n[1/3] Ridge回归 (2阶多项式)...")
    models['Ridge'] = RidgeRegressor()
    models['Ridge'].train(df_sample)
    
    print("\n[2/3] 多项式回归 (3阶)...")
    models['Poly3'] = PolynomialRegressor(degree=3)
    models['Poly3'].train(df_sample)
    
    print("\n[3/3] SVR (支持向量回归)...")
    models['SVR'] = SVRRegressor()
    models['SVR'].train(df_sample)
    
    print("\n" + "="*70)
    print("  Step 4: 交叉验证 (5折时序分割)")
    print("="*70)
    
    cv_results = {}
    for name, model in models.items():
        print(f"\n[交叉验证] {name}...")
        fold_results, metrics = model.cross_validate(df_sample, n_splits=5)
        cv_results[name] = metrics
    
    print("\n" + "="*70)
    print("  Step 5: 表1.1 填报结果 (3模型对比)")
    print("="*70)
    target_x = [7.132, 18.526, 84.337, 123.554, 167.667]
    
    all_preds = {}
    for name, model in models.items():
        df_pred = processor.get_prediction_features(target_x, df_sample)
        preds = model.predict(df_pred)
        all_preds[name] = preds
    
    print("\n" + "="*70)
    print(f"{'校正前 x':<12} | {'Ridge':<10} | {'Poly3':<10} | {'SVR':<10}")
    print("-" * 70)
    
    for i, x_val in enumerate(target_x):
        row = f"{x_val:<15} | "
        row += f"{all_preds['Ridge'][i]:<13.3f} | "
        row += f"{all_preds['Poly3'][i]:<13.3f} | "
        row += f"{all_preds['SVR'][i]:<13.3f}"
        print(row)
    print("-" * 70)
    
    print("\n" + "="*70)
    print("  Step 6: 模型性能对比")
    print("="*70)
    
    print("\n" + "="*100)
    print(f"{'模型':<15} | {'训练RMSE':<10} | {'训练MAE':<10} | {'训练R²':<10} | {'CV RMSE':<10} | {'CV R²':<10} | {'评价':<15}")
    print("-" * 100)
    
    for name, model in models.items():
        rmse = model.train_rmse
        mae = model.train_mae
        r2 = model.train_r2
        cv_rmse = cv_results[name]['mean_rmse']
        cv_r2 = cv_results[name]['mean_r2']
        
        if cv_r2 > 0.5:
            rating = "✅ 泛化良好"
        elif cv_r2 > 0:
            rating = "⚠️ 可接受"
        else:
            rating = "❌ 过拟合"
        
        print(f"{name:<15} | {rmse:<12.4f} | {mae:<12.4f} | {r2:<12.4f} | {cv_rmse:<12.4f} | {cv_r2:<12.4f} | {rating:<15}")
    print("-" * 100)
    
    best_model = max(models.keys(), key=lambda x: cv_results[x]['mean_r2'])
    print(f"\n【最优模型】{best_model} (CV R²={cv_results[best_model]['mean_r2']:.4f})")
    
    print("\n" + "="*70)
    print("  Step 7: 生成学术图表 (每张图独立)")
    print("="*70)
    visualizer = Q1Visualizer()
    time_series = df_sample['Time']
    
    ridge_preds = models['Ridge'].predict(df_sample)
    poly3_preds = models['Poly3'].predict(df_sample)
    best_preds = models[best_model].predict(df_sample)
    
    print("\n[图1] CEEMDAN分解图...")
    visualizer.plot_ceemdan(time_series, df_sample['A'], imfs, a_smooth, 
                           save_path="output/01_ceemdan.png")
    
    print("\n[图2-1] 漂移校正主图...")
    visualizer.plot_drift_main(time_series, df_sample['A'], df_sample['B'], best_preds,
                               save_path="output/02_drift_main.png")
    
    print("\n[图2-2] 误差对比图...")
    visualizer.plot_error_comparison(time_series, df_sample['A'], df_sample['B'], best_preds,
                                     save_path="output/02_error_comparison.png")
    
    print("\n[图2-3] 局部放大图...")
    visualizer.plot_drift_zoom(time_series, df_sample['A'], df_sample['B'], best_preds,
                               save_path="output/02_drift_zoom.png")
    
    print("\n[图3] 残差热力图...")
    visualizer.plot_residual_heatmap(df_sample['A'].values, df_sample['B'].values, best_preds,
                                     save_path="output/03_residual_heatmap.png")
    
    print("\n[图3-2] 校正量分布...")
    visualizer.plot_calibration_amount(df_sample['A'].values, best_preds,
                                       save_path="output/03_calibration_amount.png")
    
    print("\n[图4-1] 原始数据对比...")
    visualizer.plot_model_comparison_original(time_series, df_sample['A'].values, df_sample['B'].values,
                                              save_path="output/04_original_comparison.png")
    
    print("\n[图4-2] 多模型对比...")
    visualizer.plot_model_comparison_results(time_series, df_sample['B'].values,
                                             ridge_preds, poly3_preds, best_preds,
                                             best_model_name=best_model,
                                             save_path="output/04_model_comparison.png")
    
    print("\n[图4-3] 局部放大...")
    visualizer.plot_model_zoom(time_series, df_sample['B'].values, df_sample['A'].values,
                               best_preds, best_model_name=best_model,
                               save_path="output/04_zoom.png")
    
    print("\n[图5-1] 误差直方图...")
    visualizer.plot_error_histogram(df_sample['A'].values, df_sample['B'].values,
                                    best_preds, best_model_name=best_model,
                                    save_path="output/05_error_histogram.png")
    
    print("\n[图5-2] 误差箱线图...")
    visualizer.plot_error_boxplot(df_sample['A'].values, df_sample['B'].values,
                                  best_preds, best_model_name=best_model,
                                  save_path="output/05_error_boxplot.png")
    
    print("\n[图5-3] 累积误差曲线...")
    visualizer.plot_error_cumulative(df_sample['A'].values, df_sample['B'].values,
                                     best_preds, best_model_name=best_model,
                                     save_path="output/05_error_cumulative.png")
    
    print("\n[图5-4] 误差vsA值...")
    visualizer.plot_error_vs_A(df_sample['A'].values, df_sample['B'].values,
                               best_preds, best_model_name=best_model,
                               save_path="output/05_error_vs_A.png")
    
    print("\n[图5-5] 误差QQ图...")
    best_error = best_preds - df_sample['B'].values
    visualizer.plot_error_qq(best_error, best_model_name=best_model,
                             save_path="output/05_error_qq.png")
    
    print("\n[图5-6] 误差时序...")
    visualizer.plot_error_timeseries(df_sample['A'].values, df_sample['B'].values,
                                     best_preds, best_model_name=best_model,
                                     save_path="output/05_error_timeseries.png")
    
    print("\n[图6-1] 校正前散点图...")
    visualizer.plot_scatter_before(df_sample['A'].values, df_sample['B'].values,
                                   save_path="output/06_scatter_before.png")
    
    print("\n[图6-2] Ridge校正后散点图...")
    visualizer.plot_scatter_ridge(ridge_preds, df_sample['B'].values,
                                  save_path="output/06_scatter_ridge.png")
    
    print("\n[图6-3] 最佳模型校正后散点图...")
    visualizer.plot_scatter_best(best_preds, df_sample['B'].values,
                                 best_model_name=best_model,
                                 save_path="output/06_scatter_best.png")
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("  管线执行完毕!")
    print("="*70)
    print(f"\n总用时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    print(f"\n输出文件:")
    print(f"  - eda_plots/     (4张EDA分析图)")
    print(f"  - output/        (6张算法结果图)")

if __name__ == "__main__":
    main()
