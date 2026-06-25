import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 文件路径列表
files = [
    '按日非0需求序列.xlsx',
    '按周非0需求序列.xlsx',
    '按月非0需求序列.xlsx',
    '按日混合序列 (2).xlsx',
    '按周混合序列 (2).xlsx',
    '按月混合序列 (2).xlsx',
    '按日需求间隔序列结果.xlsx',
    '按周需求间隔序列结果.xlsx',
    '按月需求间隔序列结果.xlsx'
]

# 分类结果表
adi_cv_file = 'ADI和CV结果表.xlsx'

# 读取分类结果表
adi_cv_df = pd.read_excel(adi_cv_file)
# 假设分类列名为'分类'，如果列名不同，请根据实际修改
if '分类结果' not in adi_cv_df.columns:
    raise ValueError("ADI和CV结果表中未找到'分类'列，请检查列名")

# 获取所有唯一分类
categories = adi_cv_df['分类结果'].unique()

# 为每个文件绘制箱线图
for file in files:
    # 读取数据
    df = pd.read_excel(file)

    # 检查列名，兼容'demand'或'需求间隔'
    demand_col = 'demand' if 'demand' in df.columns else '需求间隔'
    if demand_col not in df.columns:
        print(f"文件 {file} 中未找到'demand'或'需求间隔'列，跳过")
        continue

    # 根据materialno合并分类结果
    merged_df = df.merge(adi_cv_df[['materialno', '分类结果']], on='materialno', how='inner')

    # 按分类分组数据
    data_by_category = [merged_df[merged_df['分类结果'] == cat][demand_col].dropna() for cat in categories]

    # 绘制箱线图
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_by_category, labels=categories, patch_artist=True)
    plt.title(f'{os.path.basename(file)} 按分类的{demand_col}箱线图')
    plt.xlabel('分类')
    plt.ylabel(demand_col)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 保存图像
    output_file = f'{os.path.splitext(file)[0]}_boxplot.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'已保存箱线图：{output_file}')