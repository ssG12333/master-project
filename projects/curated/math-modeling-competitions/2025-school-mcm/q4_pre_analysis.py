import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据，假设附件1为CSV文件
# 请根据实际文件路径替换 'Attachment 1.csv'
df = pd.read_csv('Attachment 1.csv')

# 调试：打印列名以确认
print(df.columns)

# 筛选指定用户，使用正确的列名 '用户ID (User ID)'
target_users = ['U10', 'U1951', 'U1833', 'U26447']
df_filtered = df[df['用户ID (User ID)'].isin(target_users)]

# 将时间列转换为日期时间格式
df_filtered['时间 (Time)'] = pd.to_datetime(df_filtered['时间 (Time)'])

# 定义时间段划分函数（每两小时一个时间段）
def get_time_period(hour):
    periods = [
        '00:00-02:00', '02:00-04:00', '04:00-06:00', '06:00-08:00',
        '08:00-10:00', '10:00-12:00', '12:00-14:00', '14:00-16:00',
        '16:00-18:00', '18:00-20:00', '20:00-22:00', '22:00-00:00'
    ]
    index = hour // 2
    return periods[index]

# 添加时间段列
df_filtered['时间段'] = df_filtered['时间 (Time)'].dt.hour.apply(get_time_period)

# 用户行为映射
behavior_map = {1: '观看', 2: '点赞', 3: '评论', 4: '关注'}
df_filtered['用户行为中文'] = df_filtered['用户行为 (User behaviour)'].map(behavior_map)

# 定义固定的颜色列表，使用Set3调色板
colors = sns.color_palette("Set3", n_colors=12)  # 确保颜色足够覆盖时间段和行为

# 为每个用户生成一张独立的图表
for user in target_users:
    user_data = df_filtered[df_filtered['用户ID (User ID)'] == user]

    # 创建画布，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 时间段占比
    time_period_counts = user_data['时间段'].value_counts()
    ax1.pie(time_period_counts, labels=time_period_counts.index, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 8}, colors=colors[:len(time_period_counts)])
    ax1.set_title(f'{user} 时间段占比')

    # 用户行为占比
    behavior_counts = user_data['用户行为中文'].value_counts()
    ax2.pie(behavior_counts, labels=behavior_counts.index, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 8}, colors=colors[:len(behavior_counts)])
    ax2.set_title(f'{user} 用户行为占比')

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f'user_behavior_pie_charts_{user}.png')
    plt.close()  # 关闭画布以释放内存

print("时序图已保存为 user_behavior_pie_charts_U*.png")