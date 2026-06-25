import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prettytable import PrettyTable
from math import sqrt
import matplotlib

matplotlib.use('TkAgg')
import joblib
import time


# 定义 series_to_supervised 函数
def series_to_supervised(data, n_in=12, n_out=1, dropnan=True):
    n_vars = data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # 输入序列 (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 输出序列 (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 数据加载和预处理
# 1. 加载数据
dataset = pd.read_csv('2023-2024.csv', header=0, index_col=0)

# 2. 填充缺失值
dataset = dataset.fillna(method='ffill').fillna(method='bfill').fillna(dataset.median())

# 3. 确保 'AQI' 列存在
if 'AQI' not in dataset.columns:
    raise KeyError("列 'AQI' 不存在，请检查数据集。")

# 4. 转换为监督学习格式：用过去12小时的数据预测下一小时的AQI
n_in = 12
reframed = series_to_supervised(dataset.values.astype('float32'), n_in=n_in, n_out=1)
print("Reframed shape:", reframed.shape)

# 5. 数据集顺序划分（前80%训练，后20%测试）
n_train = int(len(reframed) * 0.8)
train = reframed.values[:n_train, :]
test = reframed.values[n_train:, :]

# 6. 只用训练集拟合scaler，防止信息泄露
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train)

# 7. 归一化
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

# 8. 分离特征和标签
n_features = dataset.shape[1]
target_col_index = dataset.columns.get_loc('AQI')
X_train = train_scaled[:, :-n_features]
y_train = train_scaled[:, -n_features + target_col_index]
X_test = test_scaled[:, :-n_features]
y_test = test_scaled[:, -n_features + target_col_index]

# 定义 BP 神经网络
bp = MLPRegressor(
    hidden_layer_sizes=(64,),
    max_iter=1,
    random_state=72,
    warm_start=True,
    verbose=False
)

# 手动实现训练过程
train_loss_curve = []

# 记录训练开始时间
start_time = time.time()

# 训练模型并记录损失
epochs = 100
for epoch in range(epochs):
    bp.fit(X_train, y_train)

    # 计算训练集损失
    train_pred = bp.predict(X_train)
    train_loss = mean_squared_error(y_train, train_pred)
    train_loss_curve.append(train_loss)

    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

# 记录训练结束时间并计算总耗时
end_time = time.time()
total_time = end_time - start_time
print(f"\n训练总耗时: {total_time:.2f} 秒")

# 可视化训练损失
plt.figure(figsize=(12, 6))
plt.plot(train_loss_curve, label='Train Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('loss_curve.png')
plt.close()

# 预测
yhat = bp.predict(X_test)

# 反归一化（构造占位数组，仅填充 AQI 列）
n_features = scaler.n_features_in_  # 保证与scaler一致
placeholder_pred = np.zeros((len(yhat), n_features))
placeholder_pred[:, target_col_index] = yhat
predicted_data = scaler.inverse_transform(placeholder_pred)[:, target_col_index].reshape(-1, 1)

placeholder_true = np.zeros((len(y_test), n_features))
placeholder_true[:, target_col_index] = y_test
true_data = scaler.inverse_transform(placeholder_true)[:, target_col_index].reshape(-1, 1)

# 绘制真实值 vs 预测值
plt.figure(figsize=(6, 3), dpi=300)
plt.plot(true_data, label='True Value', color='blue', linewidth=0.5)
plt.plot(predicted_data, label='Prediction', color='red', linewidth=0.5)

# 设置图表标题及字体大小
plt.title('AQI Prediction vs True Value', fontsize=6)

# 设置x轴和y轴标签及字体大小
plt.xlabel('Time Step', fontsize=5)
plt.ylabel('AQI', fontsize=5)

# 设置图例，并调整图例字体大小
plt.legend(prop={'size': 4})

# 调整x轴和y轴刻度标签的字体大小
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)

plt.grid(True)  # 主网格线
plt.tight_layout()
plt.savefig('aqi_prediction_result.png')  # 保存图像
plt.close()

# 保存预测结果到CSV文件
results_df = pd.DataFrame({
    'True_Value': true_data.flatten(),
    'Predicted_Value': predicted_data.flatten()
})
results_df.to_csv('model_bp_prediction_results.csv', index=False)
print("预测结果已保存至: model_bp_prediction_results.csv")


# 评估指标
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100


mse = mean_squared_error(true_data, predicted_data)
rmse = sqrt(mse)
mae = mean_absolute_error(true_data, predicted_data)
mape_value = mape(true_data, predicted_data)
r2 = r2_score(true_data, predicted_data)

# 输出评估表格
table = PrettyTable(['Metric', 'Value'])
table.add_row(['MSE', f"{mse:.4f}"])
table.add_row(['RMSE', f"{rmse:.4f}"])
table.add_row(['MAE', f"{mae:.4f}"])
table.add_row(['MAPE', f"{mape_value:.2f}%"])
table.add_row(['R² Score', f"{r2:.4f}"])
print(table)

# 保存模型到文件
joblib.dump(bp, 'bp_model.pkl')
print("BP模型已保存至: bp_model.pkl")

# 保存 scaler 到文件
joblib.dump(scaler, 'scaler.pkl')
print("scaler已保存至: scaler.pkl")


# 加载推理数据
try:
    inference_data = pd.read_csv('inference.csv', header=0, index_col=0)
except FileNotFoundError:
    print("错误：inference.csv 文件未找到。请确保文件存在。")
    exit()

# 填充缺失值 (与训练数据处理方式保持一致)
inference_data = inference_data.fillna(method='ffill').fillna(method='bfill').fillna(inference_data.median())

# 确保推理数据包含所有训练时的特征列
missing_cols = set(dataset.columns) - set(inference_data.columns)
for c in missing_cols:
    inference_data[c] = 0  # 或者用其他合适的值填充

inference_data = inference_data[dataset.columns]  # 确保列顺序一致

# 确保 'AQI' 列存在
if 'AQI' not in inference_data.columns:
    raise KeyError("推理数据中列 'AQI' 不存在，请检查数据集。")

# 获取推理的初始 12 小时数据
# 我们需要从 inference_data 中获取最后 n_in (12) 个时间步的数据，以便进行首次预测
# 并且这些数据应该是与训练数据具有相同结构（所有特征列）
last_12_hours_raw = inference_data.iloc[-n_in:].values.astype('float32')

# 初始化预测列表
predicted_aqi_inference = []

# 定义需要预测的未来小时数（一周 = 7天 * 24小时 = 168小时）
future_hours_to_predict = 168

# 获取 AQI 列的索引
aqi_col_index_inference = dataset.columns.get_loc('AQI')

# 逐步预测未来数据
current_input_sequence = last_12_hours_raw

for i in range(future_hours_to_predict):
    # 将当前输入序列归一化
    # 注意：这里需要创建一个完整的12小时监督学习格式的行来使用 scaler
    # 由于 scaler 是在 reframed 数据上拟合的，它期望一个包含12个时间步所有特征的行
    # 这里我们只关心预测下一个值，所以我们构建一个形状为 (1, n_in * n_features) 的数组

    # 将 current_input_sequence 展平为一维数组，然后 reshape 为 (1, -1)
    # 确保只有一行的12个时间步的特征数据

    # 假设 current_input_sequence 已经是 (n_in, n_features)
    # 我们需要将其转化为 (1, n_in * n_features) 来符合 scaler 预期
    temp_reshaped_input = current_input_sequence.reshape(1, -1)

    # 创建一个占位符，用于反归一化。因为scaler.transform是针对整个训练集X做的，
    # 而inference的X_inference是只针对了X的特征部分
    # 因此，我们先构建一个完整的包含12*features个特征的行，然后进行归一化
    # 接着，再取反归一化后的AQI部分

    # 构建一个虚拟的 reframed 行，用于归一化
    # 这需要将 last_12_hours_raw 和一个占位符的未来数据合并
    # 实际上，我们只需要对输入特征进行归一化

    # scaler期望的输入是 (样本数, 特征数)
    # 对于 inference，我们只有一个样本（当前的 12 小时序列）
    # 但这个样本是由 n_in * n_features 组成的
    scaled_input_sequence = scaler.transform(np.concatenate([current_input_sequence.flatten(), np.zeros(n_features)]).reshape(1, -1))

    # 提取特征部分（不包括目标列）
    X_inference_step = scaled_input_sequence[:, :-n_features]

    # 进行预测
    next_aqi_scaled = bp.predict(X_inference_step)[0]

    # 反归一化预测值
    placeholder_next_pred = np.zeros((1, n_features))
    placeholder_next_pred[0, aqi_col_index_inference] = next_aqi_scaled
    next_aqi_true = scaler.inverse_transform(placeholder_next_pred)[0, aqi_col_index_inference]
    predicted_aqi_inference.append(next_aqi_true)

    # 更新 current_input_sequence，用于下一次预测
    # 移除最旧的一小时数据，并添加新的预测值（作为最新的一小时数据）
    # 这里我们需要构建一个新的行来代表所有特征，即使我们只预测了AQI
    # 为了保持 current_input_sequence 的形状为 (n_in, n_features)

    # 创建一个包含所有特征的新行，AQI是预测值，其他特征可以是来自 inference_data 的最后一个时间步的数据
    new_hour_features = inference_data.iloc[-1].values.astype('float32').copy()
    new_hour_features[aqi_col_index_inference] = next_aqi_true  # 将预测的AQI放入

    current_input_sequence = np.vstack((current_input_sequence[1:], new_hour_features))

print(f"成功预测未来 {future_hours_to_predict} 小时的 AQI。")

# 加载 23-24 年真实数据作为对比
# 需要确保 '2023-2024.csv' 文件中的时间戳是连续的，且覆盖了推理预测的未来一周
# 这里我们假设推理是从 dataset 的末尾开始的
# 实际对比时，您需要根据 inference.csv 的起始时间来截取 dataset 中的真实数据

# 假设推理是从训练集结束后的第一个时间点开始的
# 提取 dataset 中与预测周期对应的真实 AQI 数据
# 为了简化，我们假设 inference.csv 是 dataset 之后的新数据，并且我们是从 dataset 的末尾开始预测
# 如果 inference.csv 有明确的时间戳，您需要使用这些时间戳来匹配 dataset 中的数据
# 这里我们假设要对比的真实数据是 dataset 中在训练/测试集划分之后，且与预测时长一致的部分

# 获取原始数据集中的 AQI 列
full_aqi_data = dataset['AQI'].values.astype('float32')

# 推理预测的起始索引应该是 dataset 的最后一个数据点之后
# 但是由于 series_to_supervised 的处理，reframed 数据的索引已经偏移
# 我们需要从原始 dataset 中提取与预测时间范围相对应的真实 AQI 值
# 如果预测是从某个特定日期开始的，最好根据日期进行筛选
# 这里简单地假设我们是从整个 dataset 的末尾取真实数据进行对比

# 获取用于对比的真实数据 (如果 dataset 足够长以包含预测的未来一周)
# 由于我们的训练/测试划分是基于 reframed 数据的，所以直接从 dataset 的末尾取可能不精确
# 更好的做法是，如果知道 inference.csv 预测的起始日期和时间，则从 dataset 中找到对应的真实数据
# 或者，如果 inference.csv 仅仅是提供了起始的 12 小时数据，而我们预测的是 dataset 之后的数据，
# 那么需要从原始 dataset 中获取预测时间范围内的真实 AQI
# 考虑到 2023-2024.csv 是原始数据，我们可以从中获取实际的 AQI 走势

# 确定预测开始的真实时间点
# 为了简化，我们假设预测开始于 2023-2024.csv 文件的最后一个记录之后
# 真实对比数据应该是从 inference_data 结束后的第一个小时开始的 future_hours_to_predict 小时数据
# 假设 inference.csv 包含了直到某个时间点的数据，我们是从该时间点之后开始预测 168 小时

# 获取 dataset 的原始索引（日期时间）
dataset_index = pd.to_datetime(dataset.index)

# 获取 inference.csv 的最后一个时间点 (假设它与 dataset 中的时间格式相同)
# 假设 inference.csv 的索引是日期时间
try:
    inference_index = pd.to_datetime(inference_data.index)
    last_inference_time = inference_index[-1]
except Exception as e:
    print(f"无法解析 inference.csv 的索引为日期时间，请检查格式。错误: {e}")
    # Fallback to simple index if datetime parsing fails
    last_inference_time = None
    print("将采用简单的索引偏移来获取真实数据，这可能不准确。")

if last_inference_time:
    # 找到 dataset 中紧随 last_inference_time 的数据
    start_time_for_true = last_inference_time + pd.Timedelta(hours=1)

    # 筛选出需要对比的真实数据
    true_aqi_for_comparison_series = dataset.loc[dataset_index >= start_time_for_true, 'AQI']

    # 截取与预测长度一致的部分
    true_aqi_for_comparison = true_aqi_for_comparison_series.head(future_hours_to_predict).values
else:
    # 如果无法解析时间戳，则简单地从 dataset 的末尾取数据进行对比
    # 这在实际应用中可能不准确，因为无法保证时间对齐
    true_aqi_for_comparison = dataset['AQI'].tail(future_hours_to_predict).values

# 绘制推理预测结果与真实值的对比图
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(true_aqi_for_comparison, label='True AQI (2023-2024.csv)', color='blue', linewidth=1)
plt.plot(predicted_aqi_inference, label='Predicted AQI (Inference)', color='red', linestyle='--', linewidth=1)
plt.title(f'AQI Inference Prediction vs True Value (Next {future_hours_to_predict} Hours)')
plt.xlabel('Time Step (Hours from start of prediction)')
plt.ylabel('AQI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('aqi_inference_prediction_comparison.png')
plt.close()

print(f"推理预测结果与真实值对比图已保存至: aqi_inference_prediction_comparison.png")

# 保存推理预测结果到CSV文件
inference_results_df = pd.DataFrame({
    'Predicted_AQI': predicted_aqi_inference,
    'True_AQI_For_Comparison': true_aqi_for_comparison[:len(predicted_aqi_inference)]  # 确保长度一致
})
inference_results_df.to_csv('model_bp_inference_results.csv', index=False)
print("推理预测结果已保存至: model_bp_inference_results.csv")

# 对推理结果进行评估（如果真实数据长度足够）
if len(true_aqi_for_comparison) >= future_hours_to_predict:
    mse_inf = mean_squared_error(true_aqi_for_comparison, predicted_aqi_inference)
    rmse_inf = sqrt(mse_inf)
    mae_inf = mean_absolute_error(true_aqi_for_comparison, predicted_aqi_inference)
    mape_inf = mape(true_aqi_for_comparison, predicted_aqi_inference)
    r2_inf = r2_score(true_aqi_for_comparison, predicted_aqi_inference)

    print("\n---")
    print("## 推理预测评估指标")
    table_inf = PrettyTable(['Metric', 'Value'])
    table_inf.add_row(['MSE', f"{mse_inf:.4f}"])
    table_inf.add_row(['RMSE', f"{rmse_inf:.4f}"])
    table_inf.add_row(['MAE', f"{mae_inf:.4f}"])
    table_inf.add_row(['MAPE', f"{mape_inf:.2f}%"])
    table_inf.add_row(['R² Score', f"{r2_inf:.4f}"])
    print(table_inf)
else:
    print("\n注意：真实数据不足以覆盖整个推理预测周期，无法对推理结果进行全面评估。")