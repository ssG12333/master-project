import pandas as pd
import numpy as np
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import xgboost as xgb
import shap
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

warnings.filterwarnings('ignore')

class Q3DataProcessor:
    """
    步骤一：负责异构时间戳对齐与时延特征(Lag)构建
    """
    def __init__(self, target_freq='1H'):
        self.target_freq = target_freq

    def align_and_merge(self, df_dict):
        """
        假设传入的是一个字典，包含不同传感器的 DataFrame
        例如: {'Rainfall': df1, 'WaterPressure': df2, ...}
        每个 df 必须有 'Time' 列
        """
        print(f"--- 正在进行多源异构数据重采样对齐 (目标频率: {self.target_freq}) ---")
        merged_df = None
        
        for name, df in df_dict.items():
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.set_index('Time')
            
            if name == 'Rainfall':
                resampled = df.resample(self.target_freq).sum()
            elif name == 'Seismic':
                resampled = df.resample(self.target_freq).sum()
            else:
                resampled = df.resample(self.target_freq).mean()
                
            resampled.columns = [name]
            
            if merged_df is None:
                merged_df = resampled
            else:
                merged_df = merged_df.join(resampled, how='outer')
                
        return merged_df

    def create_lag_features(self, df):
        """
        构建岩土工程中的滞后特征 (Time-Lag) 与累积特征
        """
        print("--- 正在构建时空滞后与累积特征 ---")
        df = df.copy()
        
        if 'Rainfall' in df.columns:
            df['Rainfall_sum_24h'] = df['Rainfall'].rolling(window=24, min_periods=1).sum()
            df['Rainfall_lag_12h'] = df['Rainfall'].shift(12)
            
        if 'WaterPressure' in df.columns:
            df['WaterPressure_lag_6h'] = df['WaterPressure'].shift(6)
            
        if 'Seismic' in df.columns:
            df['Seismic_lag_2h'] = df['Seismic'].shift(2)
            
        return df

class ImputerAndDetector:
    """
    步骤二：负责 MissForest 插补与孤立森林(Isolation Forest)异常提取
    """
    def __init__(self, contamination=0.02):
        self.contamination = contamination

    def impute_missing_values(self, df, target_col=None):
        """
        使用基于随机森林的 IterativeImputer 实现多变量联合插补 (MissForest)
        target_col: 如果是测试集，指定目标列（该列全为NaN，不参与插补，后续单独预测）
        """
        print("--- 正在执行 MissForest 多变量时空联合插补 ---")
        times = df.index
        columns = df.columns
        
        # 如果指定了 target_col 且该列全为 NaN，则先排除它
        exclude_col = None
        df_to_impute = df.copy()
        if target_col is not None and target_col in df.columns:
            if df[target_col].isna().all():
                exclude_col = target_col
                df_to_impute = df.drop(columns=[target_col])
        
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=42),
                                   max_iter=5, random_state=42)
        
        imputed_data = imputer.fit_transform(df_to_impute)
        df_imputed = pd.DataFrame(imputed_data, columns=df_to_impute.columns, index=times)
        
        # 如果排除了目标列，重新添加回去
        if exclude_col is not None:
            df_imputed[exclude_col] = df[exclude_col]
            # 确保列的顺序与原始 DataFrame 一致
            df_imputed = df_imputed[columns]
        
        return df_imputed

    def detect_anomalies(self, df, features):
        """
        单变量异常检测与共同异常提取
        """
        print("--- 正在执行基于孤立森林的异常联合检测 ---")
        anomaly_matrix = pd.DataFrame(index=df.index)
        
        for feat in features:
            iso = IsolationForest(contamination=self.contamination, random_state=42)
            preds = iso.fit_predict(df[[feat]])
            anomaly_matrix[f'{feat}_Anomaly'] = (preds == -1).astype(int)
            
        anomaly_matrix['Joint_Anomaly_Count'] = anomaly_matrix.sum(axis=1)
        anomaly_matrix['Is_Joint_Anomaly'] = (anomaly_matrix['Joint_Anomaly_Count'] >= 2).astype(int)
        
        return anomaly_matrix

class ShapEvaluator:
    """
    步骤三：负责构建 XGBoost 回归并提取 SHAP 贡献度
    """
    def __init__(self, target_col='SurfaceDisp'):
        self.target_col = target_col
        self.model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
        
    def train_and_evaluate(self, df, feature_cols):
        """训练回归预测模型"""
        print("--- 正在训练 XGBoost 位移预测模型 ---")
        df_clean = df.dropna(subset=feature_cols + [self.target_col])
        X = df_clean[feature_cols]
        y = df_clean[self.target_col]
        
        self.model.fit(X, y)
        r2 = self.model.score(X, y)
        print(f"[完成] 模型训练完成, 训练集 R2: {r2:.4f}")
        return X, y
        
    def calculate_shap(self, X):
        """利用 SHAP 博弈论计算特征边际贡献度"""
        print("--- 正在计算 SHAP 全局归因贡献度 ---")
        
        # 定义包装预测函数，避免 XGBoost 版本兼容性问题
        def predict_fn(X_data):
            if hasattr(X_data, 'values'):
                return self.model.predict(X_data.values)
            return self.model.predict(X_data)
        
        # 使用 KernelExplainer 绕过兼容性问题
        print("   -> 使用 KernelExplainer 计算 SHAP 值...")
        background = shap.sample(X, 50, random_state=42)
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X, nsamples=200)
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        total_shap = mean_abs_shap.sum()
        importance_pct = (mean_abs_shap / total_shap) * 100
        
        pct_dict = {col: pct for col, pct in zip(X.columns, importance_pct)}
        return shap_values, explainer, pct_dict

class DynamicWeightEnv(gym.Env):
    """
    改进版动态特征赋权环境：
    1. 扩展状态空间包含历史上下文
    2. 改进奖励函数加入权重变化惩罚和趋势预测奖励
    3. 使用滑动窗口机制
    """
    def __init__(self, X, y, window_size=24):
        super().__init__()
        self.X = X.values
        self.y = y.values
        self.n_samples, self.n_features = self.X.shape
        self.window_size = window_size
        self.current_step = 0
        
        # 动作空间：每个特征的权重调整
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(self.n_features,), dtype=np.float32)
        
        # 状态空间：当前特征 + 过去window_size步特征均值 + 特征变化率 + 上一时刻权重
        obs_dim = self.n_features * 3 + self.n_features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # 标准化
        self.X_mean = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0) + 1e-8
        self.X_norm = (self.X - self.X_mean) / self.X_std
        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y) + 1e-8
        self.y_norm = (self.y - self.y_mean) / self.y_std
        
        self.prev_weights = np.ones(self.n_features) / self.n_features
        self.prev_pred = 0.0
        
    def _get_observation(self):
        idx = self.current_step
        # 当前特征
        current = self.X_norm[idx]
        
        # 过去window_size步的均值
        start_idx = max(0, idx - self.window_size)
        historical_mean = np.mean(self.X_norm[start_idx:idx+1], axis=0)
        
        # 特征变化率（当前 vs 历史均值）
        change_rate = current - historical_mean
        
        # 上一时刻权重
        obs = np.concatenate([current, historical_mean, change_rate, self.prev_weights])
        return obs.astype(np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_weights = np.ones(self.n_features) / self.n_features
        self.prev_pred = 0.0
        return self._get_observation(), {}
        
    def step(self, action):
        # Softmax 转换为权重
        weights = np.exp(action) / np.sum(np.exp(action))
        
        # 加权预测
        y_pred = np.sum(weights * self.X_norm[self.current_step])
        
        # 基础奖励：预测误差
        error = y_pred - self.y_norm[self.current_step]
        base_reward = -float(error ** 2)
        
        # 权重平滑惩罚：防止权重剧烈变化
        weight_change = np.sum(np.abs(weights - self.prev_weights))
        smooth_penalty = -0.1 * weight_change
        
        # 趋势奖励：如果预测方向正确，给予额外奖励
        trend_reward = 0.0
        if self.current_step > 0:
            actual_trend = self.y_norm[self.current_step] - self.y_norm[self.current_step - 1]
            pred_trend = y_pred - self.prev_pred
            if actual_trend * pred_trend > 0:  # 同向
                trend_reward = 0.5
        
        reward = base_reward + smooth_penalty + trend_reward
        
        self.prev_weights = weights.copy()
        self.prev_pred = y_pred
        self.current_step += 1
        
        terminated = bool(self.current_step >= self.n_samples)
        truncated = False
        
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape[0], dtype=np.float32)
        info = {'weights': weights}
        
        return obs, reward, terminated, truncated, info

class DRLWeightingAgent:
    """
    改进版：使用 PPO 算法替代 DDPG，训练更稳定
    """
    def __init__(self):
        self.model = None
        
    def train_and_extract_weights(self, X, y, timesteps=10000):
        print("--- 正在训练 DRL 动态赋权智能体 (PPO) ---")
        env = DynamicWeightEnv(X, y, window_size=24)
        
        # 使用 PPO 替代 DDPG，更稳定
        self.model = PPO("MlpPolicy", env, learning_rate=3e-4, 
                        n_steps=2048, batch_size=64, n_epochs=10,
                        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                        verbose=0, seed=42)
        
        # 训练并记录奖励
        reward_history = []
        from stable_baselines3.common.callbacks import BaseCallback
        
        class RewardLoggerCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.rewards = []
            def _on_step(self) -> bool:
                if 'rollout/ep_rew_mean' in self.logger.name_to_value:
                    self.rewards.append(self.logger.name_to_value['rollout/ep_rew_mean'])
                return True
        
        callback = RewardLoggerCallback()
        self.model.learn(total_timesteps=timesteps, callback=callback)
        reward_history = callback.rewards
        
        print("--- 正在提取随时间变化的特征动态贡献度 ---")
        obs, _ = env.reset()
        weights_history = []
        
        for i in range(len(X)):
            action, _ = self.model.predict(obs, deterministic=True)
            weights = np.exp(action) / np.sum(np.exp(action))
            weights_history.append(weights)
            
            next_idx = min(i + 1, len(X) - 1)
            env.current_step = next_idx
            if next_idx < len(X):
                obs = env._get_observation()
            else:
                obs = np.zeros(env.observation_space.shape[0], dtype=np.float32)
            
        return np.array(weights_history), reward_history