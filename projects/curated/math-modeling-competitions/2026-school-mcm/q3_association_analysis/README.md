# 问题 3：多源异构数据插补与多维关联分析

## 问题定义

五类传感器（降雨量、孔隙水压力、微震事件、爆破振动、表面位移）的时间戳和采样频率不统一，且存在缺失值。需要：
1. 异构时间戳对齐 + 时滞特征构建
2. 缺失值填补（MissForest 迭代插补）
3. 联合异常检测（IsolationForest + PPO 智能体）
4. 多变量关联权重分析（XGBoost + SHAP）

## 代码架构

### Q3DataProcessor (`core_models.py`)

```python
class Q3DataProcessor:
    def align_and_merge(self, df_dict):
        """
        异构时间戳对齐策略:
        - Rainfall/Seismic: resample('1H').sum()   (累积型)
        - 其他传感器:        resample('1H').mean()  (均值型)
        - 统一 join outer → 含 NaN 的完整时间表
        """

    def create_lag_features(self, df):
        """
        岩土工程时滞特征:
        - Rainfall_sum_24h:     24h 累计降雨 (渗透滞后)
        - Rainfall_lag_12h:     12h 前降雨
        - Rainfall_lag_24h:     24h 前降雨
        - Rainfall_lag_48h:     48h 前降雨
        - Rainfall_lag_72h:     72h 前降雨 (深层渗透)
        - PorePressure_diff:    孔压一阶差分 (瞬时响应)
        - PorePressure_lag_6h:  6h 前孔压
        """
```

### Q3Modeler (`core_models.py`)

```python
class Q3Modeler:
    def impute_missing(self, df):
        """
        MissForest: IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100),
            max_iter=50,
            random_state=42
        )
        每次迭代用其他列预测缺失列，直至收敛
        """

    def detect_anomalies(self, df):
        """
        双重方案:
        1. IsolationForest(contamination=0.05) 无监督异常检测
        2. PPO 智能体 (Gymnasium 自定义环境):
           - State: 多变量滑动窗口
           - Action: {0:正常, 1:异常}
           - Reward: 与 IsolationForest 的 F1 一致性
        """

    def analyze_association(self, df):
        """
        XGBoost 训练 → SHAP 分析:
        - shap_beeswarm:   全局特征贡献分布
        - shap_dependence: Top3 特征依赖曲线
        - shap_importance: 特征重要性排序
        """
```

### PPO 异常检测环境 (`core_models.py`)

```python
class AnomalyDetectionEnv(gym.Env):
    """将异常检测形式化为序列决策问题"""
    def __init__(self, data, labels=None):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, n_features))
        self.action_space = spaces.Discrete(2)  # 0:正常, 1:异常

    def step(self, action):
        # reward = +1 if action matches IsolationForest label, -1 otherwise
        # done when reaches end of sequence
```

## 输出图表（20 张）

### EDA 系列

| 多变量时序趋势 | 变量分布 | 相关矩阵 |
|:---:|:---:|:---:|
| ![trends](figures/eda_timeseries_trends.png) | ![dist](figures/eda_variable_distributions.png) | ![corr](figures/eda_correlation_matrix.png) |

| 缺失值热力图 | 降雨-位移时滞分析 | 孔压-位移散点 |
|:---:|:---:|:---:|
| ![missing](figures/eda_missing_values_heatmap.png) | ![lag](figures/eda_rainfall_surface_lag.png) | ![scatter](figures/eda_waterpressure_surface_scatter.png) |

### 预处理系列

| 填补前后对比 | 异常检测结果 |
|:---:|:---:|
| ![impute](figures/preprocess_imputation_comparison.png) | ![anomaly](figures/preprocess_anomaly_detection.png) |

### SHAP 关联分析

| 特征重要性排序 | SHAP Beeswarm | Top3 依赖曲线 |
|:---:|:---:|:---:|
| ![importance](figures/shap_feature_importance_bar.png) | ![beeswarm](figures/shap_beeswarm.png) | ![dependence](figures/shap_dependence_top3.png) |

### 验证与测试

| 数据划分 | 预测 vs 实际 | 残差直方图 |
|:---:|:---:|:---:|
| ![split](figures/validation_data_split.png) | ![scatter](figures/validation_pred_vs_actual_scatter.png) | ![hist](figures/validation_residual_histogram.png) |

| 时序预测对比 | 测试集预测 |
|:---:|:---:|
| ![comparison](figures/train_val_test_prediction_comparison.png) | ![test](figures/test_prediction_scatter.png) |

## 运行方式

```bash
pip install pandas numpy scikit-learn xgboost shap gymnasium stable-baselines3 matplotlib seaborn
cd q3_association_analysis
python main.py
```
