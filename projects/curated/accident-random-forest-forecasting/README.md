# 随机森林交通事故预测

## 项目简介

本项目围绕交通事故数量预测，使用 `RandomForestRegressor` 构建按日事故量预测模型。完整管线包括事故 Excel 数据清洗、类别特征比例编码、时间序列补齐 (2022-01-01 至 2023-08-31)、随机森林回归建模、评估指标计算和 12 类可视化输出。

## 技术栈

- **核心**: Python 3.8+, pandas, NumPy
- **机器学习**: scikit-learn (RandomForestRegressor, LabelEncoder, metrics)
- **可视化**: matplotlib, seaborn
- **数据格式**: .xlsx (原始), .csv (中间), .xlsx (预测结果)

---

## 完整管线 (`forecast_accidents.py`)

脚本按 7 个步骤顺序执行，输出目录为 `accident_time_series_output_2023_aug_daily_rf/`。

### 步骤 1: 数据读取

```python
data = pd.read_excel('2022-2023.xlsx')
# 更正列名: 'wheather_condition' -> 'weather_condition'
data = data.rename(columns={'wheather_condition': 'weather_condition'})
```

- **原始数据**: 2022-2023 年交通事故 Excel 记录
- **列名修正**: 自动更正拼写错误的 `wheather_condition`

### 步骤 2: 数据预处理

```python
# 日期解析 (支持多种格式)
data['date'] = pd.to_datetime(data['inverse_data'], errors='coerce')
data = data.dropna(subset=['date'])

# 分类变量缺失值填充
categorical_cols = ['weather_condition', 'road_delineation', 'road_direction', 'road_type']
for col in categorical_cols:
    data[col] = data[col].fillna('未知')

# 时间特征提取
data['weekday'] = data['date'].dt.dayofweek   # 0=Monday, 6=Sunday
data['month'] = data['date'].dt.month          # 1-12
data['day'] = data['date'].dt.day              # 1-31
```

- 缺失日期记录被移除
- 缺失的分类变量填充为 `'未知'`
- 添加 3 个时间特征: `weekday`, `month`, `day`

### 步骤 3: 生成按天汇总时间序列

```python
# 完整日期范围
dates = pd.date_range(start='2022-01-01', end='2023-08-31', freq='D')

# 按天统计事故数量
daily_data = data.groupby('date').size().reset_index(name='accident_count')

# 类别变量比例编码 (基于历史比例加权)
# 训练数据: < 2023-08-01
# 当天有数据 -> 使用当天实际类别分布加权
# 当天无数据 -> 使用训练集历史比例加权
```

**类别特征编码方法**: 对每个类别变量，计算当天各类别的出现比例，然后用比例作为权重计算加权平均值作为当天的数值特征。这种方法保留了类别分布的统计信息，避免了 one-hot 编码带来的维度膨胀。

```python
# 合并完整日期序列 (含缺失日期补零)
daily_data = full_df.merge(daily_data, on='date', how='left')
daily_data['accident_count'] = daily_data['accident_count'].fillna(0)
```

**最终时间序列特征**:

| 特征 | 类型 | 说明 |
|------|------|------|
| `date` | datetime | 日期 |
| `accident_count` | int | 当日事故数量 (目标变量) |
| `weather_condition` | float | 天气状况比例编码值 |
| `road_delineation` | float | 道路标线比例编码值 |
| `road_direction` | float | 道路方向比例编码值 |
| `road_type` | float | 道路类型比例编码值 |
| `weekday` | int | 星期几 (0-6) |
| `month` | int | 月份 (1-12) |
| `day` | int | 日期 (1-31) |

### 步骤 4: 训练/测试集划分

```python
train_data = daily_data[daily_data['date'] < '2023-08-01']  # 2022-01-01 ~ 2023-07-31
test_data = daily_data[(daily_data['date'] >= '2023-08-01') & (daily_data['date'] <= '2023-08-31')]
```

- **训练集**: 2022-01-01 至 2023-07-31 (约 577 天)
- **测试集**: 2023-08-01 至 2023-08-31 (31 天)
- **特征列**: `categorical_cols + ['weekday', 'month', 'day']` (共 7 维)

### 步骤 5: 模型训练与预测

```python
# 初始化随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 特征矩阵
feature_cols = categorical_cols + ['weekday', 'month', 'day']
X_train = train_data[feature_cols]
y_train = train_data['accident_count']
X_test = test_data[feature_cols]
y_test = test_data['accident_count']

# 训练
rf_model.fit(X_train, y_train)

# 预测
train_pred = rf_model.predict(X_train)
test_pred = rf_model.predict(X_test)
```

**RandomForestRegressor 超参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_estimators` | 100 | 决策树数量 |
| `random_state` | 42 | 随机种子 (保证可重现) |
| 其他参数 | 默认值 | max_depth=None, min_samples_split=2, min_samples_leaf=1 |

### 步骤 6: 评估指标

**训练集拟合指标** (从代码中计算):

```python
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_mae = mean_absolute_error(y_train, train_pred)
train_r2 = r2_score(y_train, train_pred)
```

**测试集指标** (`results/metrics.txt` 实际数值):

| 指标 | 值 | 说明 |
|------|-----|------|
| RMSE | 16.02 | 均方根误差 |
| MAE | 12.64 | 平均绝对误差 |
| R2 | 0.68 | 决定系数 (68% 方差被模型解释) |

### 步骤 7: 结果保存

```python
output_data.to_excel(os.path.join(output_dir, '2023年8月事故数量预测.xlsx'), index=False)
```

- 预测结果: `2023年8月事故数量预测.xlsx` (含日期、类别特征、实际值、预测值、误差)

---

## 可视化输出 (共 12 张图表)

| 文件 | 类型 | 说明 |
|------|------|------|
| `缺失值统计.png` | 柱状图 | 各列缺失值比例 |
| `原始事故数量时间序列.png` | 折线图 | 2022-2023 原始日事故量 |
| `填充后时间序列.png` | 折线图 | 缺失日期补零后的完整序列 |
| `{col}_分布.png` x4 | 柱状图 | 天气/道路标线/方向/类型的分布 |
| `训练集拟合.png` | 折线图 | 训练集实际 vs 预测 |
| `测试集实际_vs_预测.png` | 折线图 | 2023年8月实际 vs 预测 |
| `预测值分布.png` | 直方图+KDE | 预测值密度分布 |
| `天气贡献.png` | 柱状图 | 不同天气的事故占比 |
| `预测误差分布.png` | 直方图+KDE | 误差分布 (有实际数据时) |
| `预测_vs_实际散点图.png` | 散点图 | 预测 vs 实际 (有实际数据时) |

---

## 工作链路

1. **读取 Excel**: `pd.read_excel('2022-2023.xlsx')` -> 列名修正
2. **清洗**: 日期解析 -> 缺失值填充 -> 时间特征 (`weekday`, `month`, `day`)
3. **聚合**: `groupby('date').size()` -> 按天统计事故数量
4. **补齐**: `pd.date_range('2022-01-01', '2023-08-31')` -> 缺失日期填充 0
5. **特征编码**: `LabelEncoder.fit_transform()` + 比例加权编码
6. **划分**: `train_data < '2023-08-01'`, `test_data >= '2023-08-01'`
7. **建模**: `RandomForestRegressor(n_estimators=100)` -> `fit()` -> `predict()`
8. **评估**: RMSE, MAE, R2
9. **输出**: 预测 Excel + 12 张可视化图表

## 关键文件

- `forecast_accidents.py`: 完整管线 (数据清洗、建模、预测、图表输出)
- `accident_time_series_output_2023_aug_daily_rf/`: 自动创建的输出目录
- `results/metrics.txt`: 测试集 RMSE=16.02, MAE=12.64, R2=0.68
- `results/daily_data.csv`: 按日聚合后的中间数据

## 整理说明

- 已移除 9MB 原始 Excel 数据、IDE 配置和冗余结果
- 当前目录保留核心脚本、指标文件和代表性图表
