import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
data_folder = 'data'
output_base_folder = 'analysis_results'
os.makedirs(output_base_folder, exist_ok=True)

# 汇总所有公司回归结果
results_summary = []

# 遍历每家公司的 Excel 文件
for file in os.listdir(data_folder):
    if file.endswith('.xlsx'):
        file_path = os.path.join(data_folder, file)
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()  # 去除列名空格等问题

        if '净资产/市值（亿元）' not in df.columns:
            print(f"{file} 缺少 '净资产/市值（亿元）' 列，跳过")
            continue

        # 提取市值
        df['市值（亿元）'] = df['净资产/市值（亿元）']

        # 时间格式处理
        df['报告日期'] = pd.to_datetime(df['报告日期'], errors='coerce')
        df['年份'] = df['报告日期'].dt.year
        df['季度'] = df['报告日期'].dt.quarter

        # 派生指标
        df['营收增长率'] = df['营业收入（亿元）'].pct_change()
        df['利润率'] = df['净利润（亿元）'] / df['营业收入（亿元）']

        # 公司名作为输出文件夹
        company_name = df['公司名称'].iloc[0]
        output_folder = os.path.join(output_base_folder, company_name)
        os.makedirs(output_folder, exist_ok=True)

        # 保存预处理数据
        df.to_excel(os.path.join(output_folder, 'processed_data.xlsx'), index=False)

        # 提取用于分析的字段
        corr_df = df[['净利润（亿元）', '市值（亿元）', '总资产（亿元）', '总负债（亿元）',
                      '每股收益（元）', '营收增长率', '利润率']]

        # 计算相关性矩阵
        corr_matrix = corr_df.corr()

        # 热力图1：每股收益相关性
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix[['每股收益（元）']].sort_values(by='每股收益（元）', ascending=False),
                    annot=True, cmap='coolwarm', fmt=".2f", cbar=True, vmin=-1, vmax=1)
        plt.title(f'{company_name}：与每股收益的相关性热力图')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, '每股收益相关热力图.png'))
        plt.close()

        # 热力图2：市值相关性
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix[['市值（亿元）']].sort_values(by='市值（亿元）', ascending=False),
                    annot=True, cmap='YlGnBu', fmt=".2f", cbar=True, vmin=-1, vmax=1)
        plt.title(f'{company_name}：与市值的相关性热力图')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, '市值相关热力图.png'))
        plt.close()

        # OLS 回归分析
        reg_df = corr_df.dropna()

        # 模型1：每股收益 为因变量
        y1 = reg_df['每股收益（元）']
        X1 = reg_df.drop(columns=['每股收益（元）'])
        X1 = sm.add_constant(X1)
        model1 = sm.OLS(y1, X1).fit()

        # 模型2：市值 为因变量
        y2 = reg_df['市值（亿元）']
        X2 = reg_df.drop(columns=['市值（亿元）'])
        X2 = sm.add_constant(X2)
        model2 = sm.OLS(y2, X2).fit()

        # 保存回归结果
        with open(os.path.join(output_folder, '回归分析结果.txt'), 'w', encoding='utf-8') as f:
            f.write(f"公司名称：{company_name}\n\n")
            f.write("【每股收益 为因变量】\n")
            f.write(model1.summary().as_text())
            f.write("\n\n【市值 为因变量】\n")
            f.write(model2.summary().as_text())

        # 汇总
        summary = {
            "公司": company_name,
            "每股收益_R2": model1.rsquared,
            "市值_R2": model2.rsquared,
            "每股收益_显著变量": ', '.join([var for var, p in model1.pvalues.items() if p < 0.05 and var != 'const']),
            "市值_显著变量": ', '.join([var for var, p in model2.pvalues.items() if p < 0.05 and var != 'const']),
        }
        results_summary.append(summary)

# 保存汇总表
summary_df = pd.DataFrame(results_summary)
summary_df.to_excel(os.path.join(output_base_folder, '整体回归分析汇总.xlsx'), index=False)

print("✅ 分析完成，所有结果保存在 analysis_results 文件夹中。")
