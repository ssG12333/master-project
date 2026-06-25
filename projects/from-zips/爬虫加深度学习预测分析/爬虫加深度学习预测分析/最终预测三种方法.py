import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示为乱码

# 数据路径
data_folder = 'data'
output_base = '最终预测三种结果'
os.makedirs(output_base, exist_ok=True)

# 模型列表
models = {
    "线性回归": LinearRegression(),
    "决策树": DecisionTreeRegressor(random_state=42),
    "随机森林": RandomForestRegressor(n_estimators=100, random_state=42)
}

# 汇总所有公司预测表现
model_scores = []
all_companies_results = []

# 设置未来5年的预测周期
forecast_years = 5
forecast_quarters = forecast_years * 4  # 5年预测，共20个季度

for file in os.listdir(data_folder):
    if not file.endswith('.xlsx'):
        continue
    path = os.path.join(data_folder, file)
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    if '净资产/市值（亿元）' not in df.columns:
        continue

    df['市值（亿元）'] = df['净资产/市值（亿元）']
    df['报告日期'] = pd.to_datetime(df['报告日期'], errors='coerce')
    df = df.sort_values('报告日期')

    # 特征构建
    df['营收增长率'] = df['营业收入（亿元）'].pct_change()
    df['利润率'] = df['净利润（亿元）'] / df['营业收入（亿元）']
    df['未来每股收益'] = df['每股收益（元）'].shift(-4)  # 预测未来一年（4季度）后的收益

    # 选取特征和标签
    features = ['净利润（亿元）', '市值（亿元）', '总资产（亿元）', '总负债（亿元）',
                '营收增长率', '利润率', '经营活动现金流净额（亿元）', '职工薪酬现金（亿元）',
                '股息支付（亿元）', '营业成本（亿元）', '营业收入（亿元）']
    if df[features + ['未来每股收益']].dropna().shape[0] < 10:
        continue  # 数据不足跳过

    data = df[features + ['未来每股收益']].dropna()
    X = data[features]
    y = data['未来每股收益']

    # 使用所有数据进行训练
    model_scores_all_data = []

    # 创建公司结果输出文件夹
    company_name = df['公司名称'].iloc[0]
    out_dir = os.path.join(output_base, company_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, '模型评估_所有数据.txt'), 'w', encoding='utf-8') as f:
        f.write(f"公司：{company_name}\n\n")
        company_results = {'公司': company_name, 'MSE': [], 'R2': [], '加权得分': 0}

        for name, model in models.items():
            model.fit(X, y)  # 使用所有数据训练模型
            future_X = X.tail(forecast_quarters)  # 获取最近的20个季度数据进行预测
            future_pred = model.predict(future_X)

            # 保存未来5年预测
            future_dates = pd.date_range(df['报告日期'].max(), periods=forecast_quarters + 1, freq='Q')[1:]
            future_df = pd.DataFrame({
                '报告日期': future_dates,
                '预测每股收益（元）': future_pred
            })

            future_df.to_excel(os.path.join(out_dir, f'{name}_未来5年预测.xlsx'), index=False)

            mse = mean_squared_error(y.tail(4), model.predict(X.tail(4)))  # 使用最后4个季度的数据计算MSE
            r2 = r2_score(y.tail(4), model.predict(X.tail(4)))  # 使用最后4个季度的数据计算R²

            f.write(f"【{name}】\n")
            f.write(f"MSE: {mse:.4f}, R²: {r2:.4f}\n\n")

            model_scores_all_data.append({
                "公司": company_name,
                "模型": name,
                "MSE": mse,
                "R2": r2
            })

            company_results['MSE'].append(mse)
            company_results['R2'].append(r2)

            # 可视化预测效果
            plt.figure(figsize=(8, 5))
            plt.plot(future_dates, future_pred, label='预测值', marker='x')
            plt.title(f'{company_name} - {name} 预测未来5年每股收益', fontsize=14)
            plt.xlabel('报告日期', fontsize=12)
            plt.ylabel('每股收益（元）', fontsize=12)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{name}_未来5年预测.png'))
            plt.close()

        # 综合评估：加权得分
        # 假设每个模型的得分权重均等，进行简单平均
        avg_mse = np.mean(company_results['MSE'])
        avg_r2 = np.mean(company_results['R2'])
        weighted_score = (avg_r2 - avg_mse)  # 可以根据实际需要调整加权策略

        company_results['加权得分'] = weighted_score
        all_companies_results.append(company_results)

# 保存所有公司模型评分汇总
score_df = pd.DataFrame(model_scores_all_data)
score_df.to_excel(os.path.join(output_base, '所有公司模型评分汇总.xlsx'), index=False)

# 保存所有公司加权得分汇总，排序按加权得分高低（推荐的选股结果）
final_df = pd.DataFrame(all_companies_results)
final_df = final_df.sort_values(by='加权得分', ascending=False)
final_df.to_excel(os.path.join(output_base, '所有公司选股预测汇总.xlsx'), index=False)

print("✅ 模型构建与评估完成，选股预测已保存到对应文件夹中。")
