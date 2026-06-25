import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os
from datetime import datetime

# 设置 matplotlib 中文字体和分辨率
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

# 创建输出目录
output_dir = 'accident_time_series_output_2023_aug_daily_rf'
os.makedirs(output_dir, exist_ok=True)

# 1. 读取数据
print("=== 步骤 1: 读取数据 ===")
data = pd.read_excel('2022-2023.xlsx')
print(f"读取的数据行数: {len(data)}")
print(f"数据列: {list(data.columns)}")
print("数据前5行:")
print(data.head())

# 更正拼写错误的列名
data = data.rename(columns={'wheather_condition': 'weather_condition'})
print("已更正列名 'wheather_condition' 为 'weather_condition'")

# 检查日期范围
print("原始数据日期范围:")
print(f"最早日期: {data['inverse_data'].min()}")
print(f"最晚日期: {data['inverse_data'].max()}")

# 可视化：缺失值统计
print("\n=== 可视化：缺失值统计 ===")
missing_data = data.isna().mean() * 100
plt.figure(figsize=(10, 6))
missing_data[missing_data > 0].sort_values().plot(kind='barh', color='#EF4444')
plt.title('各列缺失值比例 (%)')
plt.xlabel('缺失值比例 (%)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '缺失值统计.png'))
plt.close()

# 2. 数据预处理
print("\n=== 步骤 2: 数据预处理 ===")
# 检查缺失值
print("各列缺失值统计:")
print(data.isna().sum())

# 转换日期列，尝试多种格式
data['date'] = pd.to_datetime(data['inverse_data'], errors='coerce')
print(f"日期列数据类型: {data['date'].dtype}")

# 检查无效日期
invalid_dates = data['date'].isna().sum()
if invalid_dates > 0:
    print(f"⚠️ inverse_data 包含 {invalid_dates} 个无效日期，将被移除")
    data = data.dropna(subset=['date'])
    print(f"移除无效日期后，数据行数: {len(data)}")

# 检查清洗后日期范围
print("清洗后数据日期范围:")
print(f"最早日期: {data['date'].min()}")
print(f"最晚日期: {data['date'].max()}")

# 检查2023年8月数据
aug_data = data[data['date'].dt.strftime('%Y-%m') == '2023-08']
print(f"2023年8月数据记录数: {len(aug_data)}")
if not aug_data.empty:
    print("2023年8月数据前5行:")
    print(aug_data.head())

# 填充分类变量缺失值
categorical_cols = ['weather_condition', 'road_delineation', 'road_direction', 'road_type']
for col in categorical_cols:
    if col in data.columns and data[col].isna().sum() > 0:
        print(f"⚠️ {col} 列包含 {data[col].isna().sum()} 个缺失值，将填充为 '未知'")
        data[col] = data[col].fillna('未知')
    elif col not in data.columns:
        print(f"⚠️ 列 {col} 不存在于数据中，将创建并填充 '未知'")
        data[col] = '未知'

# 添加时间特征
data['weekday'] = data['date'].dt.dayofweek  # 星期几（0-6）
data['month'] = data['date'].dt.month  # 月份
data['day'] = data['date'].dt.day  # 日期
print(f"已添加时间特征，数据行数: {len(data)}")

# 可视化：原始事故数量时间序列
print("\n=== 可视化：原始事故数量时间序列 ===")
daily_counts = data.groupby('date').size().reset_index(name='accident_count')
print(f"原始时间序列行数: {len(daily_counts)}")
print("原始时间序列前5行:")
print(daily_counts.head())
plt.figure(figsize=(12, 6))
plt.plot(daily_counts['date'], daily_counts['accident_count'], color='#3B82F6')
plt.title('原始事故数量时间序列 (2022-2023)')
plt.xlabel('日期')
plt.ylabel('事故数量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '原始事故数量时间序列.png'))
plt.close()

# 可视化：协变量分布
print("\n=== 可视化：协变量分布 ===")
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=col, order=data[col].value_counts().index, palette='viridis')
    plt.title(f'{col} 分布')
    plt.xlabel(col)
    plt.ylabel('计数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{col}_分布.png'))
    plt.close()

# 3. 生成按天汇总的时间序列（按比例融合特征）
print("\n=== 步骤 3: 生成按天汇总的时间序列 ===")
# 获取所有日期
dates = pd.date_range(start='2022-01-01', end='2023-08-31', freq='D')
full_df = pd.DataFrame({'date': dates})
full_df['date'] = pd.to_datetime(full_df['date'])
print(f"完整时间序列日期数: {len(full_df)}")

# 按天汇总事故数量
daily_data = data.groupby('date').size().reset_index(name='accident_count')

# 计算历史协变量比例（训练数据：2022-01-01至2023-07-31）
train_data_raw = data[data['date'] < '2023-08-01']
prop_dict = {}
label_encoders = {}
for col in categorical_cols:
    # 计算比例
    value_counts = train_data_raw[col].value_counts(normalize=True)
    prop_dict[col] = value_counts.to_dict()
    print(f"{col} 历史比例:")
    for k, v in prop_dict[col].items():
        print(f"  {k}: {v:.4f}")

    # 编码类别
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# 按天融合协变量（按比例加权）
daily_features = []
for date in daily_data['date']:
    day_data = data[data['date'] == date]
    row = {'date': date}
    for col in categorical_cols:
        if not day_data.empty:
            # 当天有数据，按当天比例加权
            value_counts = day_data[col].value_counts(normalize=True)
            weighted_value = sum(k * v for k, v in value_counts.items())
        else:
            # 当天无数据，使用历史比例加权
            weighted_value = sum(k * v for k, v in prop_dict[col].items())
        row[col] = weighted_value
    daily_features.append(row)
daily_features = pd.DataFrame(daily_features)

# 合并事故数量和特征
daily_data = daily_data.merge(daily_features, on='date', how='left')

# 合并到完整时间序列
daily_data = full_df.merge(daily_data, on='date', how='left')
daily_data['accident_count'] = daily_data['accident_count'].fillna(0)
for col in categorical_cols:
    if daily_data[col].isna().any():
        # 填充缺失值（使用历史比例加权平均）
        daily_data[col] = daily_data[col].fillna(sum(k * v for k, v in prop_dict[col].items()))

# 添加时间特征到时间序列
daily_data['weekday'] = daily_data['date'].dt.dayofweek
daily_data['month'] = daily_data['date'].dt.month
daily_data['day'] = daily_data['date'].dt.day
print(f"按天时间序列数据行数: {len(daily_data)}")
print("时间序列前5行:")
print(daily_data.head())
print("时间序列日期范围:")
print(f"最早日期: {daily_data['date'].min()}")
print(f"最晚日期: {daily_data['date'].max()}")

# 检查2023年8月时间序列
aug_daily = daily_data[daily_data['date'].dt.strftime('%Y-%m') == '2023-08']
print(f"时间序列中2023年8月数据行数: {len(aug_daily)}")
print("2023年8月时间序列前5行:")
print(aug_daily.head())

# 保存时间序列数据以供检查
daily_data.to_csv(os.path.join(output_dir, 'daily_data.csv'), index=False)
print(f"时间序列数据已保存至: {os.path.join(output_dir, 'daily_data.csv')}")

# 可视化：填充后的时间序列
print("\n=== 可视化：填充后的时间序列 ===")
plt.figure(figsize=(12, 6))
plt.plot(daily_data['date'], daily_data['accident_count'], color='#3B82F6')
plt.title('填充后的时间序列 (2022-2023)')
plt.xlabel('日期')
plt.ylabel('事故数量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '填充后时间序列.png'))
plt.close()

# 4. 准备训练和测试数据
print("\n=== 步骤 4: 准备训练和测试数据 ===")
train_data = daily_data[daily_data['date'] < '2023-08-01']
test_data = daily_data[(daily_data['date'] >= '2023-08-01') & (daily_data['date'] <= '2023-08-31')]
print(f"训练集数据行数: {len(train_data)}")
print(f"测试集数据行数: {len(test_data)}")
print("测试集数据预览:")
print(test_data.head())
print(f"测试集 accident_count 缺失值: {test_data['accident_count'].isna().sum()}")

# 强制生成测试集
if test_data.empty:
    print("⚠️ 错误：测试集为空，强制生成2023年8月数据框架")
    test_data = pd.DataFrame({
        'date': pd.date_range(start='2023-08-01', end='2023-08-31', freq='D')
    })
    test_data['date'] = pd.to_datetime(test_data['date'])
    for col in categorical_cols:
        test_data[col] = sum(k * v for k, v in prop_dict[col].items())
    test_data['accident_count'] = 0
    test_data['weekday'] = test_data['date'].dt.dayofweek
    test_data['month'] = test_data['date'].dt.month
    test_data['day'] = test_data['date'].dt.day
    print(f"生成测试集框架，行数: {len(test_data)}")
elif test_data['accident_count'].isna().all():
    print("⚠️ 警告：测试集 accident_count 全为 NaN，将填充为0以绘制实际 vs 预测图")
    test_data['accident_count'] = test_data['accident_count'].fillna(0)

# 5. 训练随机森林模型并预测
print("\n=== 步骤 5: 训练随机森林模型并预测 ===")
# 准备训练数据
feature_cols = categorical_cols + ['weekday', 'month', 'day']
X_train = train_data[feature_cols]
y_train = train_data['accident_count']
X_test = test_data[feature_cols]
y_test = test_data['accident_count']

# 检查训练数据完整性
if X_train.isna().any().any():
    print("⚠️ 训练特征中存在缺失值，填充为0")
    X_train = X_train.fillna(0)
if y_train.isna().any():
    print("⚠️ 训练目标值中存在缺失值，填充为0")
    y_train = y_train.fillna(0)
if X_test.isna().any().any():
    print("⚠️ 测试特征中存在缺失值，填充为0")
    X_test = X_test.fillna(0)

# 初始化随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
try:
    rf_model.fit(X_train, y_train)
    print("✅ 随机森林模型训练成功")
except Exception as e:
    print(f"❌ 随机森林模型训练失败，错误信息: {e}")
    raise

# 预测训练集
train_pred = rf_model.predict(X_train)

# 可视化：训练集拟合
print("\n=== 可视化：训练集拟合 ===")
plt.figure(figsize=(12, 6))
plt.plot(train_data['date'], y_train, label='实际事故数量', color='#3B82F6')
plt.plot(train_data['date'], train_pred, label='预测事故数量', color='#F59E0B')
plt.title('训练集实际 vs 预测事故数量 (2022-2023)')
plt.xlabel('日期')
plt.ylabel('事故数量')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '训练集拟合.png'))
plt.close()

# 计算训练集拟合指标
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_mae = mean_absolute_error(y_train, train_pred)
train_r2 = r2_score(y_train, train_pred)
print(f"训练集 RMSE: {train_rmse:.2f}")
print(f"训练集 MAE: {train_mae:.2f}")
print(f"训练集 R²: {train_r2:.2f}")

# 预测测试集
try:
    test_pred = rf_model.predict(X_test)
    print("✅ 测试集预测成功")
except Exception as e:
    print(f"❌ 测试集预测失败，错误信息: {e}")
    raise

# 检查预测结果
print("测试集预测结果预览:")
print(pd.DataFrame({'date': test_data['date'], '预测事故数量': test_pred}).head())

# 保存预测结果
output_data = test_data[['date', 'accident_count']].copy()
output_data['yhat'] = test_pred
output_data = output_data.rename(columns={'date': '日期', 'accident_count': '实际事故数量', 'yhat': '预测事故数量'})
output_data['预测误差'] = output_data['实际事故数量'] - output_data['预测事故数量']
for col in categorical_cols:
    output_data[col] = test_data[col]
print(f"预测数据行数: {len(output_data)}")
print("预测数据前5行:")
print(output_data.head())

# 6. 计算测试集评估指标
print("\n=== 步骤 6: 计算测试集评估指标 ===")
has_actual_data = not output_data['实际事故数量'].isna().all() and output_data['实际事故数量'].notna().sum() > 0 and not output_data['实际事故数量'].eq(0).all()
if has_actual_data:
    valid_data = output_data.dropna(subset=['实际事故数量', '预测事故数量'])
    if not valid_data.empty:
        rmse = np.sqrt(mean_squared_error(valid_data['实际事故数量'], valid_data['预测事故数量']))
        mae = mean_absolute_error(valid_data['实际事故数量'], valid_data['预测事故数量'])
        r2 = r2_score(valid_data['实际事故数量'], valid_data['预测事故数量'])

        with open(os.path.join(output_dir, '测试集评估指标.txt'), 'w', encoding='utf-8') as f:
            f.write(f'测试集 RMSE（均方根误差）：{rmse:.2f}\n')
            f.write(f'测试集 MAE（平均绝对误差）：{mae:.2f}\n')
            f.write(f'测试集 R²（决定系数）：{r2:.2f}\n')

        print(f'测试集 RMSE（均方根误差）：{rmse:.2f}')
        print(f'测试集 MAE（平均绝对误差）：{mae:.2f}')
        print(f'测试集 R²（决定系数）：{r2:.2f}')
    else:
        print("⚠️ 测试集实际数据无效（全为0或NaN），无法计算评估指标")
else:
    print("⚠️ 没有2023年8月的有效实际事故数据（全为0或NaN），无法计算测试集评估指标")

# 可视化：测试集实际 vs 预测
print("\n=== 可视化：测试集实际 vs 预测 ===")
plt.figure(figsize=(12, 6))
plt.plot(output_data['日期'], output_data['预测事故数量'], label='预测事故数量', color='#F59E0B')
plt.plot(output_data['日期'], output_data['实际事故数量'].fillna(0), label='实际事故数量', color='#3B82F6')
if output_data['实际事故数量'].isna().all() or output_data['实际事故数量'].eq(0).all():
    print("⚠️ 警告：实际事故数量全为0或NaN，图表可能仅显示预测值")
plt.title('测试集实际 vs 预测事故数量（2023年8月）')
plt.xlabel('日期')
plt.ylabel('事故数量')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '测试集实际_vs_预测.png'))
plt.close()

# 可视化：预测值分布
print("\n=== 可视化：预测值分布 ===")
plt.figure(figsize=(10, 6))
sns.histplot(output_data['预测事故数量'], bins=20, kde=True, color='#F59E0B')
plt.title('2023年8月预测事故数量分布')
plt.xlabel('预测事故数量')
plt.ylabel('频率')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '预测值分布.png'))
plt.close()

# 可视化：协变量贡献（以 weather_condition 为例）
print("\n=== 可视化：协变量贡献 ===")
weather_effect = data.groupby('weather_condition').size().reset_index(name='accident_count')
weather_effect['accident_count'] /= weather_effect['accident_count'].sum()
plt.figure(figsize=(10, 6))
sns.barplot(x='weather_condition', y='accident_count', data=weather_effect, palette='viridis')
plt.title('不同天气类型的事故占比')
plt.xlabel('天气类型')
plt.ylabel('事故占比')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '天气贡献.png'))
plt.close()

# 可视化：预测误差分布（如果有实际数据）
if has_actual_data:
    valid_data = output_data.dropna(subset=['实际事故数量', '预测事故数量'])
    if not valid_data.empty:
        print("\n=== 可视化：预测误差分布 ===")
        plt.figure(figsize=(10, 6))
        sns.histplot(valid_data['预测误差'], bins=20, kde=True, color='#EF4444')
        plt.title('测试集预测误差分布（2023年8月）')
        plt.xlabel('预测误差')
        plt.ylabel('频率')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '预测误差分布.png'))
        plt.close()

# 可视化：预测 vs 实际散点图（如果有实际数据）
if has_actual_data:
    valid_data = output_data.dropna(subset=['实际事故数量', '预测事故数量'])
    if not valid_data.empty:
        print("\n=== 可视化：预测 vs 实际散点图 ===")
        plt.figure(figsize=(8, 8))
        plt.scatter(valid_data['实际事故数量'], valid_data['预测事故数量'], color='#3B82F6', alpha=0.6)
        plt.plot([valid_data['实际事故数量'].min(), valid_data['实际事故数量'].max()],
                 [valid_data['实际事故数量'].min(), valid_data['实际事故数量'].max()],
                 color='#EF4444', linestyle='--')
        plt.title('测试集预测 vs 实际事故数量（2023年8月）')
        plt.xlabel('实际事故数量')
        plt.ylabel('预测事故数量')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '预测_vs_实际散点图.png'))
        plt.close()

# 7. 保存预测结果
print("\n=== 步骤 7: 保存预测结果 ===")
output_columns = ['日期', 'weather_condition', 'road_delineation', 'road_direction', 'road_type', '实际事故数量', '预测事故数量', '预测误差']
output_data[output_columns].to_excel(os.path.join(output_dir, '2023年8月事故数量预测.xlsx'), index=False)
print(f"✅ 成功生成预测表格：{os.path.join(output_dir, '2023年8月事故数量预测.xlsx')}")

print(f"✅ 所有输出已保存至目录：{output_dir}")