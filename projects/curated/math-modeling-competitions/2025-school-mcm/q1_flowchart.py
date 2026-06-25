import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'  # 使用黑体字体，根据系统实际情况调整
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建一个图形和坐标轴
fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.axis('off')

# 定义每个步骤的文本和位置
steps = [
    {
        "text": "数据准备：\n1. 读取数据并转换日期类型\n2. 排序数据\n3. 生成全量索引和数据框\n4. 合并数据并计算 next_day_follows\n5. 统计数据集信息并保存",
        "x": 2,
        "y": 11
    },
    {
        "text": "序列生成：\n1. 定义 create_sequences 函数\n2. 针对每位博主生成特征序列和目标值\n3. 检查是否生成序列，若否提示错误并终止",
        "x": 4,
        "y": 8
    },
    {
        "text": "数据集划分与标准化：\n1. 划分训练集、验证集和测试集\n2. 标准化特征数据\n3. 转换为张量并创建 Dataset 和 DataLoader",
        "x": 6,
        "y": 11
    },
    {
        "text": "模型训练：\n1. 定义模型并移动到设备上\n2. 定义损失函数和优化器\n3. 使用学习率调度器和早停法\n4. 记录训练和验证损失并保存\n5. 绘制损失曲线",
        "x": 8,
        "y": 8
    },
    {
        "text": "模型评估：\n1. 在测试集上计算均方误差损失\n2. 计算平均相对误差和平均绝对误差\n3. 保存评估结果",
        "x": 10,
        "y": 11
    },
    {
        "text": "预测与结果展示：\n1. 定义 create_prediction_sequences 函数\n2. 生成预测序列并标准化\n3. 使用模型进行预测\n4. 整理预测结果并保存",
        "x": 8,
        "y": 5
    },
    {
        "text": "回归分析可视化：\n1. 绘制散点图和理想预测线\n2. 保存可视化结果和数据",
        "x": 6,
        "y": 2
    }
]

# 定义好看的颜色
box_color = '#E5F6FF'  # 矩形框填充颜色
edge_color = '#73A6FF'  # 矩形框边框颜色
arrow_color = '#FF9900'  # 箭头颜色
text_color = '#333333'  # 文本颜色

# 绘制每个步骤的矩形框和文本
for step in steps:
    rect = patches.Rectangle((step["x"] - 1.5, step["y"] - 2), 3, 4, linewidth=2,
                             edgecolor=edge_color, facecolor=box_color, alpha=0.8)
    ax.add_patch(rect)
    ax.text(step["x"], step["y"], step["text"], ha='center', va='center',
            fontsize=10, color=text_color)

# 绘制箭头表示流程
connections = [
    (steps[0], steps[1]),
    (steps[1], steps[2]),
    (steps[2], steps[3]),
    (steps[3], steps[4]),
    (steps[3], steps[5]),
    (steps[5], steps[6])
]

for start, end in connections:
    ax.annotate("",
                xy=(end["x"], end["y"]),
                xytext=(start["x"], start["y"]),
                arrowprops=dict(arrowstyle="->", color=arrow_color, linewidth=2))

# 保存图片
plt.savefig('flowchart.png')
plt.show()