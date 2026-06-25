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
output_base = '最终预测的回测'
os.makedirs(output_base, exist_ok=True)

# 模型列表
models = {
    "线性回归": LinearRegression(),
    "决策树": DecisionTreeRegressor(random_state=42),
    "随机森林": RandomForestRegressor(n_estimators=100, random_state=42)
}

# 汇总所有公司预测表现
model_scores = []

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

    # 拆分训练测试集：前80%训练，后20%测试
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 创建公司结果输出文件夹
    company_name = df['公司名称'].iloc[0]
    out_dir = os.path.join(output_base, company_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, '模型评估.txt'), 'w', encoding='utf-8') as f:
        f.write(f"公司：{company_name}\n\n")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            f.write(f"【{name}】\n")
            f.write(f"MSE: {mse:.4f}, R²: {r2:.4f}\n\n")

            model_scores.append({
                "公司": company_name,
                "模型": name,
                "MSE": mse,
                "R2": r2
            })

            # 可视化预测效果
            plt.figure(figsize=(8, 5))
            plt.plot(y_test.values, label='真实值', marker='o')
            plt.plot(y_pred, label='预测值', marker='x')
            plt.title(f'{company_name} - {name} 预测未来每股收益', fontsize=14)
            plt.xlabel('测试样本序号', fontsize=12)
            plt.ylabel('每股收益（元）', fontsize=12)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{name}_预测效果.png'))
            plt.close()

# 保存所有模型评分
score_df = pd.DataFrame(model_scores)
score_df.to_excel(os.path.join(output_base, '所有公司模型评分汇总.xlsx'), index=False)

print("✅ 模型构建与评估完成，结果保存在 最终预测的回测 文件夹中。")