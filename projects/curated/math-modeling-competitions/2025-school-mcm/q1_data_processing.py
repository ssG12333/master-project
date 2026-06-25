import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
import uuid

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取实际数据CSV文件
df_actual = pd.read_csv('Attachment 1.csv')
# 读取预测数据CSV文件
df_predicted_raw = pd.read_csv('所有.csv')

# 转换实际数据的时间列为日期格式
df_actual['时间 (Time)'] = pd.to_datetime(df_actual['时间 (Time)'])
df_actual['Date'] = df_actual['时间 (Time)'].dt.date

# 筛选实际数据中关注行为（用户行为=4）
follow_actual = df_actual[df_actual['用户行为 (User behaviour)'] == 4]

# 处理预测数据（仅用于曲线图）
# 假设预测数据是7月21日的
predicted_date = date(2024, 7, 21)
df_predicted = pd.DataFrame({
    '博主ID (Blogger ID)': df_predicted_raw['blogger_id'],
    'Date': [predicted_date] * len(df_predicted_raw),
    'PredictedFollows': df_predicted_raw['predicted_new_follows']
})

# 为后续图表准备合并数据（不包含预测数据）
follow_df = follow_actual.copy()

# 1. 每日新增关注数热力图（无预测数据）
def plot_daily_follow_heatmap():
    pivot_data = follow_df.groupby(['Date', '博主ID (Blogger ID)']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt='d')
    plt.title('每日新增关注数热力图')
    plt.xlabel('博主ID')
    plt.ylabel('日期')
    plt.savefig('daily_follow_heatmap.png')
    plt.close()

# 2. 每个博主的每日关注数曲线图（添加7月21日预测数据）
def plot_blogger_follow_curves():
    # 按日期和博主ID统计实际关注数
    daily_follows_actual = follow_df.groupby(['Date', '博主ID (Blogger ID)']).size().unstack(fill_value=0)

    # 获取所有博主ID
    all_bloggers = list(set(daily_follows_actual.columns).union(set(df_predicted['博主ID (Blogger ID)'])))

    plt.figure(figsize=(12, 8))
    for blogger in all_bloggers:
        # 绘制实际数据
        if blogger in daily_follows_actual.columns:
            actual_data = daily_follows_actual[blogger]
            line, = plt.plot(actual_data.index, actual_data, linestyle='-', label=f'博主 {blogger}')
            line_color = line.get_color()  # 获取实际数据的颜色
        else:
            line_color = None  # 如果没有实际数据，稍后分配颜色

        # 绘制预测数据（仅7月21日）
        predicted_row = df_predicted[df_predicted['博主ID (Blogger ID)'] == blogger]
        if not predicted_row.empty:
            predicted_value = predicted_row['PredictedFollows'].iloc[0]
            # 获取实际数据的最后一天（如果有），用于连接
            if blogger in daily_follows_actual.columns:
                last_actual_date = daily_follows_actual.index[-1]
                last_actual_value = daily_follows_actual[blogger].iloc[-1]
                # 连接最后一天到预测数据
                plt.plot([last_actual_date, predicted_date], [last_actual_value, predicted_value],
                         linestyle='--', color=line_color)
            # 单独绘制预测点
            plt.scatter([predicted_date], [predicted_value], marker='o', s=100,
                       color=line_color, label=None if blogger in daily_follows_actual.columns else f'博主 {blogger}')

    plt.title('每个博主每日关注数曲线图（实际+7月21日预测）')
    plt.xlabel('日期')
    plt.ylabel('新增关注数')
    # 调整图例：缩小字体，增加列数，放置在图表外部右侧
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    plt.xticks(rotation=45)
    # 调整布局，确保图例有足够空间
    plt.subplots_adjust(right=0.75)
    plt.savefig('blogger_follow_curves.png')
    plt.close()

# 3. 新增关注时序热力图（无预测数据）
def plot_follow_timeseries_heatmap():
    follow_df['Hour'] = follow_df['时间 (Time)'].dt.hour
    pivot_data = follow_df.groupby(['Date', 'Hour']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, cmap='YlGnBu', annot=True, fmt='d')
    plt.title('新增关注时序热力图')
    plt.xlabel('小时')
    plt.ylabel('日期')
    plt.savefig('follow_timeseries_heatmap.png')
    plt.close()

# 4. 新增关注数量每日柱状图（无预测数据）
def plot_daily_follow_bar():
    daily_counts = follow_df.groupby('Date').size()

    plt.figure(figsize=(12, 8))
    daily_counts.plot(kind='bar')
    plt.title('新增关注数量每日柱状图')
    plt.xlabel('日期')
    plt.ylabel('新增关注数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('daily_follow_bar.png')
    plt.close()

# 执行所有绘图函数
plot_daily_follow_heatmap()
plot_blogger_follow_curves()
plot_follow_timeseries_heatmap()
plot_daily_follow_bar()