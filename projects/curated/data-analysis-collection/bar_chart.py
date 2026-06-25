import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置支持中文的字体
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取 Excel 文件
# 假设文件名为 "ADI和CV结果表.xlsx"，请确保文件路径正确
df = pd.read_excel("ADI和CV结果表.xlsx")

# 统计分类结果的分布
category_counts = df['分类结果'].value_counts()

# 创建柱状图
plt.figure(figsize=(10, 6))
bars = category_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.title('分类结果分布柱状图', fontsize=14)
plt.xlabel('分类结果', fontsize=12)
plt.ylabel('数量', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 在柱子上方添加数值标签
for bar in bars.patches:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # x 坐标：柱子中心
        height + 0.5,  # y 坐标：柱子顶部稍上方
        int(height),  # 显示整数值
        ha='center',  # 水平居中
        va='bottom',  # 垂直底部对齐
        fontsize=10
    )

# 显示图形
plt.tight_layout()
plt.show()