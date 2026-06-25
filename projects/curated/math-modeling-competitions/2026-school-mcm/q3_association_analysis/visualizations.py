import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

class Q3Visualizations:
    def __init__(self, output_dir='.'):
        self.output_dir = output_dir

    # ========== 步骤1: EDA 探索性分析 (7张图) ==========

    def plot_eda_timeseries(self, df):
        fig, axes = plt.subplots(5, 1, figsize=(14, 18))
        variables = ['Rainfall', 'WaterPressure', 'Seismic', 'DeepDisp', 'SurfaceDisp']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
        for i, (var, ax, color) in enumerate(zip(variables, axes, colors)):
            if var in df.columns:
                ax.plot(df.index, df[var], color=color, linewidth=0.8, alpha=0.8)
                ax.set_ylabel(var, fontsize=13)
                ax.tick_params(labelsize=11)
                if i < 4:
                    ax.set_xticklabels([])
        axes[-1].set_xlabel('时间', fontsize=13)
        fig.suptitle('图1.1 五变量时间序列趋势图', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/eda_timeseries_trends.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: eda_timeseries_trends.png")

    def plot_eda_missing_heatmap(self, df):
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(df.isnull().T, cmap='YlOrRd', cbar=True, ax=ax)
        ax.set_ylabel('变量', fontsize=13)
        ax.set_xlabel('时间索引', fontsize=13)
        ax.tick_params(labelsize=11)
        fig.suptitle('图1.2 缺失值分布热力图', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/eda_missing_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: eda_missing_values_heatmap.png")

    def plot_eda_correlation_matrix(self, df):
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1,
                    square=True, linewidths=0.5, ax=ax, annot_kws={'fontsize': 12})
        ax.tick_params(labelsize=12)
        fig.suptitle('图1.3 变量相关性矩阵热力图', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/eda_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: eda_correlation_matrix.png")

    def plot_eda_variable_distributions(self, df):
        variables = ['Rainfall', 'WaterPressure', 'Seismic', 'DeepDisp', 'SurfaceDisp']
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        for i, var in enumerate(variables):
            if var in df.columns:
                sns.histplot(df[var].dropna(), kde=True, color='#2E86AB', ax=axes[i], bins=50)
                axes[i].set_title(var, fontsize=13)
                axes[i].tick_params(labelsize=11)
        axes[-1].axis('off')
        fig.suptitle('图1.4 各变量分布直方图与KDE曲线', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/eda_variable_distributions.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: eda_variable_distributions.png")

    def plot_eda_rainfall_deepdisp_lag(self, df):
        if 'Rainfall' in df.columns and 'DeepDisp' in df.columns:
            fig, ax1 = plt.subplots(figsize=(14, 6))
            ax1.fill_between(df.index, df['Rainfall'], color='#2E86AB', alpha=0.4, label='降雨量')
            ax1.set_ylabel('降雨量 (mm)', fontsize=13, color='#2E86AB')
            ax1.tick_params(axis='y', labelcolor='#2E86AB', labelsize=11)
            ax2 = ax1.twinx()
            ax2.plot(df.index, df['DeepDisp'], color='#C73E1D', linewidth=1.2, label='深部位移')
            ax2.set_ylabel('深部位移 (mm)', fontsize=13, color='#C73E1D')
            ax2.tick_params(axis='y', labelcolor='#C73E1D', labelsize=11)
            ax1.set_xlabel('时间', fontsize=13)
            ax1.tick_params(labelsize=11)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
            fig.suptitle('图1.5 降雨量与深部位移滞后响应图', fontsize=16, y=0.99)
            fig.savefig(f'{self.output_dir}/eda_rainfall_deepdisp_lag.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("-> 已保存: eda_rainfall_deepdisp_lag.png")

    def plot_eda_rainfall_surface_lag(self, df):
        if 'Rainfall' in df.columns and 'SurfaceDisp' in df.columns:
            fig, ax1 = plt.subplots(figsize=(14, 6))
            ax1.fill_between(df.index, df['Rainfall'], color='#F18F01', alpha=0.4, label='降雨量')
            ax1.set_ylabel('降雨量 (mm)', fontsize=13, color='#F18F01')
            ax1.tick_params(axis='y', labelcolor='#F18F01', labelsize=11)
            ax2 = ax1.twinx()
            ax2.plot(df.index, df['SurfaceDisp'], color='#A23B72', linewidth=1.2, label='表面位移')
            ax2.set_ylabel('表面位移 (mm)', fontsize=13, color='#A23B72')
            ax2.tick_params(axis='y', labelcolor='#A23B72', labelsize=11)
            ax1.set_xlabel('时间', fontsize=13)
            ax1.tick_params(labelsize=11)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
            fig.suptitle('图1.6 降雨量与表面位移滞后响应图', fontsize=16, y=0.99)
            fig.savefig(f'{self.output_dir}/eda_rainfall_surface_lag.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("-> 已保存: eda_rainfall_surface_lag.png")

    def plot_eda_waterpressure_surface_scatter(self, df):
        if 'WaterPressure' in df.columns and 'SurfaceDisp' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.regplot(x='WaterPressure', y='SurfaceDisp', data=df, scatter_kws={'alpha': 0.4, 's': 30},
                        line_kws={'color': '#C73E1D', 'linewidth': 2}, ax=ax)
            ax.set_xlabel('孔隙水压力 (kPa)', fontsize=13)
            ax.set_ylabel('表面位移 (mm)', fontsize=13)
            ax.tick_params(labelsize=12)
            fig.suptitle('图1.7 孔隙水压力与表面位移关系图', fontsize=16, y=0.99)
            fig.savefig(f'{self.output_dir}/eda_waterpressure_surface_scatter.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("-> 已保存: eda_waterpressure_surface_scatter.png")

    # ========== 步骤2: 数据预处理可视化 (2张图) ==========

    def plot_preprocess_imputation_comparison(self, df_original, df_imputed):
        variables = ['Rainfall', 'WaterPressure', 'DeepDisp', 'SurfaceDisp']
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        for i, var in enumerate(variables):
            if var in df_original.columns and var in df_imputed.columns:
                axes[i].plot(df_original.index, df_original[var], 'o', markersize=3, alpha=0.5, label='原始数据', color='#A23B72')
                axes[i].plot(df_imputed.index, df_imputed[var], '-', linewidth=0.8, label='插补后', color='#2E86AB')
                axes[i].set_title(var, fontsize=13)
                axes[i].legend(fontsize=11)
                axes[i].tick_params(labelsize=11)
        fig.suptitle('图2.1 MissForest 插补前后对比图', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/preprocess_imputation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: preprocess_imputation_comparison.png")

    def plot_preprocess_anomaly_detection(self, df_imputed, anomaly_matrix, features):
        fig, axes = plt.subplots(len(features), 1, figsize=(14, 3*len(features)))
        if len(features) == 1:
            axes = [axes]
        for i, feat in enumerate(features):
            anomaly_col = f'{feat}_Anomaly'
            if anomaly_col in anomaly_matrix.columns:
                normal_mask = anomaly_matrix[anomaly_col] == 0
                anomaly_mask = anomaly_matrix[anomaly_col] == 1
                axes[i].plot(df_imputed.index[normal_mask], df_imputed.loc[normal_mask, feat],
                            '.', markersize=3, alpha=0.6, label='正常', color='#2E86AB')
                axes[i].plot(df_imputed.index[anomaly_mask], df_imputed.loc[anomaly_mask, feat],
                            'o', markersize=5, label='异常', color='#C73E1D')
                axes[i].set_title(feat, fontsize=13)
                axes[i].legend(fontsize=11)
                axes[i].tick_params(labelsize=11)
        fig.suptitle('图2.2 孤立森林异常检测结果', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/preprocess_anomaly_detection.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: preprocess_anomaly_detection.png")

    # ========== 步骤3: 模型验证 (6-4划分) (5张图) ==========

    def plot_validation_data_split(self, train_idx, val_idx, time_index):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(time_index[train_idx], np.ones(len(train_idx)), '|', markersize=15,
               markeredgewidth=2, label='训练集 (60%)', color='#2E86AB')
        ax.plot(time_index[val_idx], np.ones(len(val_idx)), '|', markersize=15,
               markeredgewidth=2, label='验证集 (40%)', color='#C73E1D')
        ax.set_yticks([])
        ax.set_xlabel('时间', fontsize=13)
        ax.legend(fontsize=12, loc='upper right')
        ax.tick_params(labelsize=11)
        fig.suptitle('图3.1 训练集与验证集时间划分示意图', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/validation_data_split.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: validation_data_split.png")

    def plot_validation_pred_vs_actual_scatter(self, y_val, y_pred):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(y_val, y_pred, alpha=0.5, s=30, color='#2E86AB')
        min_val, max_val = min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线 (y=x)')
        ax.set_xlabel('实际值', fontsize=13)
        ax.set_ylabel('预测值', fontsize=13)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        from sklearn.metrics import r2_score
        r2 = r2_score(y_val, y_pred)
        ax.text(0.05, 0.95, f'R2 = {r2:.4f}', transform=ax.transAxes, fontsize=14,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        fig.suptitle('图3.2 验证集预测值 vs 实际值散点图', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/validation_pred_vs_actual_scatter.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: validation_pred_vs_actual_scatter.png")

    def plot_validation_timeseries_comparison(self, time_index_val, y_val, y_pred):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(time_index_val, y_val, '-', linewidth=1.2, label='实际值', color='#2E86AB')
        ax.plot(time_index_val, y_pred, '--', linewidth=1.2, label='预测值', color='#C73E1D')
        ax.set_xlabel('时间', fontsize=13)
        ax.set_ylabel('表面位移 (mm)', fontsize=13)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=11)
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        textstr = f'R2 = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        fig.suptitle('图3.3 验证集预测值与实际值时间序列对比', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/validation_timeseries_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: validation_timeseries_comparison.png")

    def plot_validation_residual_histogram(self, y_val, y_pred):
        residuals = y_val - y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='#2E86AB', bins=50, ax=ax)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='均值线')
        ax.set_xlabel('残差 (实际值 - 预测值)', fontsize=13)
        ax.set_ylabel('频数', fontsize=13)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        from scipy import stats
        _, p_value = stats.shapiro(residuals[:5000])
        ax.text(0.02, 0.98, f'Shapiro-Wilk 正态性检验 p-value = {p_value:.4f}',
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        fig.suptitle('图3.4 验证集残差分布直方图', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/validation_residual_histogram.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: validation_residual_histogram.png")

    def plot_validation_residual_vs_pred(self, y_pred, y_val):
        residuals = y_val - y_pred
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(y_pred, residuals, alpha=0.4, s=20, color='#2E86AB')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('预测值', fontsize=13)
        ax.set_ylabel('残差 (实际值 - 预测值)', fontsize=13)
        ax.tick_params(labelsize=12)
        fig.suptitle('图3.5 残差 vs 预测值散点图', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/validation_residual_vs_pred.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: validation_residual_vs_pred.png")

    # ========== 步骤4: SHAP 归因分析 (3张图) ==========

    def plot_shap_beeswarm(self, shap_values, X):
        fig, ax = plt.subplots(figsize=(12, 8))
        import shap as shap_lib
        shap_lib.summary_plot(shap_values, X, show=False, plot_size=None)
        ax.set_title('图4.1 SHAP 特征边际贡献度全局解释图 (蜂拥图)', fontsize=16, pad=20)
        fig.savefig(f'{self.output_dir}/shap_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: shap_beeswarm.png")

    def plot_shap_feature_importance_bar(self, pct_dict):
        sorted_items = sorted(pct_dict.items(), key=lambda x: x[1], reverse=True)
        features = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = sns.color_palette('viridis', len(features))
        bars = ax.barh(features, values, color=colors)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}%', va='center', fontsize=12)
        ax.set_xlabel('贡献度百分比 (%)', fontsize=13)
        ax.set_ylabel('特征', fontsize=13)
        ax.tick_params(labelsize=12)
        fig.suptitle('图4.2 SHAP 特征重要性条形图', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/shap_feature_importance_bar.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: shap_feature_importance_bar.png")

    def plot_shap_dependence_top3(self, shap_values, X, pct_dict):
        sorted_items = sorted(pct_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_features = [item[0] for item in sorted_items]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        import shap as shap_lib
        for i, feat in enumerate(top3_features):
            feat_idx = list(X.columns).index(feat)
            shap_lib.dependence_plot(feat_idx, shap_values, X, show=False, ax=axes[i])
            axes[i].set_title(feat, fontsize=13)
            axes[i].tick_params(labelsize=11)
        fig.suptitle('图4.3 SHAP 依赖关系图 (Top 3 特征)', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/shap_dependence_top3.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: shap_dependence_top3.png")

    # ========== 步骤5: 实验集预测 (2张图) ==========

    def plot_train_test_prediction_comparison(self, train_time, train_actual, train_pred,
                                              val_time, val_actual, val_pred,
                                              test_time, test_pred):
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))

        # 上子图: 训练集 - 实际值 vs 预测值
        axes[0].plot(train_time, train_actual, '-', linewidth=1.0, alpha=0.7, label='训练集实际值', color='#2E86AB')
        axes[0].plot(train_time, train_pred, '--', linewidth=1.0, alpha=0.8, label='训练集预测值', color='#1B4F72')
        axes[0].set_ylabel('表面位移 (mm)', fontsize=13)
        axes[0].legend(fontsize=11, loc='upper left')
        axes[0].tick_params(labelsize=11)
        from sklearn.metrics import r2_score
        train_r2 = r2_score(train_actual, train_pred)
        axes[0].text(0.02, 0.95, f'训练集 R2 = {train_r2:.4f}', transform=axes[0].transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='#D5F5E3', alpha=0.7))

        # 中子图: 验证集(回测) - 实际值 vs 预测值
        axes[1].plot(val_time, val_actual, '-', linewidth=1.0, alpha=0.7, label='验证集实际值', color='#C73E1D')
        axes[1].plot(val_time, val_pred, '--', linewidth=1.0, alpha=0.8, label='验证集预测值', color='#922B21')
        axes[1].set_ylabel('表面位移 (mm)', fontsize=13)
        axes[1].legend(fontsize=11, loc='upper left')
        axes[1].tick_params(labelsize=11)
        val_r2 = r2_score(val_actual, val_pred)
        axes[1].text(0.02, 0.95, f'验证集(回测) R2 = {val_r2:.4f}', transform=axes[1].transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.7))

        # 下子图: 实验集预测
        axes[2].plot(test_time, test_pred, '-', linewidth=1.2, label='实验集预测值', color='#F18F01')
        axes[2].fill_between(test_time, test_pred, alpha=0.3, color='#F18F01')
        axes[2].set_xlabel('时间', fontsize=13)
        axes[2].set_ylabel('预测表面位移 (mm)', fontsize=13)
        axes[2].legend(fontsize=11, loc='upper left')
        axes[2].tick_params(labelsize=11)

        fig.suptitle('图5.1 训练集-验证集(回测)-实验集预测结果对比', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/train_val_test_prediction_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: train_val_test_prediction_comparison.png")

    def plot_test_prediction_boxplot(self, y_predicted):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=y_predicted, color='#2E86AB', ax=ax)
        ax.set_xlabel('预测表面位移 (mm)', fontsize=13)
        ax.tick_params(labelsize=12)
        stats_text = f'均值: {np.mean(y_predicted):.2f}\n中位数: {np.median(y_predicted):.2f}\n标准差: {np.std(y_predicted):.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        fig.suptitle('图5.2 实验集预测值分布箱线图', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/test_prediction_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: test_prediction_boxplot.png")

    def plot_test_prediction_scatter(self, time_index, y_predicted):
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.scatter(range(len(time_index)), y_predicted, alpha=0.6, s=30, color='#A23B72')
        ax.plot(range(len(time_index)), y_predicted, '-', linewidth=0.8, color='#A23B72', alpha=0.8)
        ax.set_xlabel('时间索引', fontsize=13)
        ax.set_ylabel('预测表面位移 (mm)', fontsize=13)
        ax.tick_params(labelsize=12)
        stats_text = f'最大值: {np.max(y_predicted):.2f} mm\n最小值: {np.min(y_predicted):.2f} mm\n均值: {np.mean(y_predicted):.2f} mm'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        fig.suptitle('图5.3 实验集表面位移预测结果散点图', fontsize=16, y=0.99)
        fig.savefig(f'{self.output_dir}/test_prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("-> 已保存: test_prediction_scatter.png")
