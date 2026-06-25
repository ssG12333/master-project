import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline


class ModelBase:
    """模型基类"""
    def __init__(self):
        self.model = None
        self.train_rmse = 0
        self.train_mae = 0
        self.train_r2 = 0
    
    def train(self, df):
        raise NotImplementedError
    
    def predict(self, df_pred):
        raise NotImplementedError
    
    def cross_validate(self, df, n_splits=5):
        raise NotImplementedError
    
    def evaluate(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2


class BaselineXGB(ModelBase):
    def __init__(self):
        super().__init__()
        self.model = xgb.XGBRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.01,
            subsample=0.5, colsample_bytree=0.5,
            reg_alpha=10.0, reg_lambda=50.0,
            min_child_weight=10,
            random_state=42
        )
        self.features = ['A_smooth', 'A_sq', 'A_cube', 'A_bin', 'A_vel']
        
    def train(self, df):
        print("\n" + "="*50)
        print("[XGBoost] 正在训练基线模型...")
        print("="*50)
        X = df[self.features]
        y = df['B']
        
        print(f"  训练样本数: {len(X)}")
        print(f"  特征数: {len(self.features)}")
        print(f"  特征列表: {self.features}")
        
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.train_rmse, self.train_mae, self.train_r2 = self.evaluate(y, y_pred)
        print(f"  [XGBoost 训练集评估] RMSE={self.train_rmse:.4f} | MAE={self.train_mae:.4f} | R2={self.train_r2:.4f}")
        return y_pred
    
    def predict(self, df_pred):
        X = df_pred[self.features]
        return self.model.predict(X)
    
    def cross_validate(self, df, n_splits=5):
        print("\n" + "="*50)
        print(f"[XGBoost] 开始 {n_splits} 折时序交叉验证...")
        print("="*50)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]
            
            X_train = df_train[self.features]
            y_train = df_train['B']
            X_test = df_test[self.features]
            y_test = df_test['B']
            
            fold_model = xgb.XGBRegressor(
                n_estimators=50, max_depth=2, learning_rate=0.01,
                subsample=0.5, colsample_bytree=0.5,
                reg_alpha=10.0, reg_lambda=50.0,
                min_child_weight=10,
                random_state=42
            )
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            
            rmse, mae, r2 = self.evaluate(y_test, y_pred)
            fold_results.append({'fold': fold, 'rmse': rmse, 'mae': mae, 'r2': r2})
            print(f"  折 {fold}/{n_splits} | 训练:{len(X_train)} | 测试:{len(X_test)} | RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")
        
        mean_rmse = np.mean([r['rmse'] for r in fold_results])
        mean_mae = np.mean([r['mae'] for r in fold_results])
        mean_r2 = np.mean([r['r2'] for r in fold_results])
        
        print(f"\n  [交叉验证均值] RMSE={mean_rmse:.4f} | MAE={mean_mae:.4f} | R2={mean_r2:.4f}")
        print("="*50)
        
        return fold_results, {'mean_rmse': mean_rmse, 'mean_mae': mean_mae, 'mean_r2': mean_r2}


class PolynomialRegressor(ModelBase):
    def __init__(self, degree=4):
        super().__init__()
        self.degree = degree
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('ridge', Ridge(alpha=1.0))
        ])
        self.features = ['A_smooth']
        
    def train(self, df):
        print("\n" + "="*50)
        print(f"[多项式回归] 正在训练 ({self.degree}阶)...")
        print("="*50)
        
        X = df[self.features].values
        y = df['B'].values
        
        print(f"  训练样本数: {len(X)}")
        print(f"  多项式阶数: {self.degree}")
        
        self.model.fit(X, y)
        
        n_features = len(self.model.named_steps['poly'].get_feature_names_out())
        print(f"  扩展特征数: {n_features}")
        y_pred = self.model.predict(X)
        self.train_rmse, self.train_mae, self.train_r2 = self.evaluate(y, y_pred)
        print(f"  [多项式回归评估] RMSE={self.train_rmse:.4f} | MAE={self.train_mae:.4f} | R2={self.train_r2:.4f}")
        return y_pred
    
    def predict(self, df_pred):
        X = df_pred[self.features].values
        return self.model.predict(X)
    
    def cross_validate(self, df, n_splits=5):
        print(f"\n[多项式回归] 开始 {n_splits} 折时序交叉验证...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]
            
            X_train = df_train[self.features].values
            y_train = df_train['B'].values
            X_test = df_test[self.features].values
            y_test = df_test['B'].values
            
            fold_model = Pipeline([
                ('poly', PolynomialFeatures(degree=self.degree)),
                ('ridge', Ridge(alpha=1.0))
            ])
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            
            rmse, mae, r2 = self.evaluate(y_test, y_pred)
            fold_results.append({'fold': fold, 'rmse': rmse, 'mae': mae, 'r2': r2})
            print(f"  折 {fold}/{n_splits} | RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")
        
        mean_rmse = np.mean([r['rmse'] for r in fold_results])
        mean_mae = np.mean([r['mae'] for r in fold_results])
        mean_r2 = np.mean([r['r2'] for r in fold_results])
        
        print(f"  [交叉验证均值] RMSE={mean_rmse:.4f} | MAE={mean_mae:.4f} | R2={mean_r2:.4f}\n")
        return fold_results, {'mean_rmse': mean_rmse, 'mean_mae': mean_mae, 'mean_r2': mean_r2}


class PolynomialRegressor(ModelBase):
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('ridge', Ridge(alpha=0.01))
        ])
        self.features = ['A']
        
    def train(self, df):
        print("\n" + "="*50)
        print(f"[多项式回归] 正在训练 ({self.degree}阶)...")
        print("="*50)
        
        X = df[self.features].values
        y = df['B'].values
        
        print(f"  训练样本数: {len(X)}")
        print(f"  多项式阶数: {self.degree}")
        print(f"  扩展特征数: {self.degree + 1}")
        print(f"  正则化参数: alpha=0.01")
        
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.train_rmse, self.train_mae, self.train_r2 = self.evaluate(y, y_pred)
        print(f"  [多项式回归评估] RMSE={self.train_rmse:.4f} | MAE={self.train_mae:.4f} | R2={self.train_r2:.4f}")
        return y_pred
    
    def predict(self, df_pred):
        X = df_pred[self.features].values
        return self.model.predict(X)
    
    def cross_validate(self, df, n_splits=5):
        print(f"\n[多项式回归] 开始 {n_splits} 折时序交叉验证...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]
            
            X_train = df_train[self.features].values
            y_train = df_train['B'].values
            X_test = df_test[self.features].values
            y_test = df_test['B'].values
            
            fold_model = Pipeline([
                ('poly', PolynomialFeatures(degree=self.degree)),
                ('ridge', Ridge(alpha=0.01))
            ])
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            
            rmse, mae, r2 = self.evaluate(y_test, y_pred)
            fold_results.append({'fold': fold, 'rmse': rmse, 'mae': mae, 'r2': r2})
            print(f"  折 {fold}/{n_splits} | RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")
        
        mean_rmse = np.mean([r['rmse'] for r in fold_results])
        mean_mae = np.mean([r['mae'] for r in fold_results])
        mean_r2 = np.mean([r['r2'] for r in fold_results])
        
        print(f"  [交叉验证均值] RMSE={mean_rmse:.4f} | MAE={mean_mae:.4f} | R2={mean_r2:.4f}\n")
        return fold_results, {'mean_rmse': mean_rmse, 'mean_mae': mean_mae, 'mean_r2': mean_r2}


class SVRRegressor(ModelBase):
    def __init__(self):
        super().__init__()
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.5))
        ])
        self.features = ['A']
        
    def train(self, df):
        print("\n" + "="*50)
        print("[SVR] 正在训练...")
        print("="*50)
        
        X = df[self.features].values
        y = df['B'].values
        
        print(f"  训练样本数: {len(X)}")
        print(f"  核函数: RBF")
        print(f"  C=10.0, epsilon=0.5")
        print(f"  gamma=auto")
        
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.train_rmse, self.train_mae, self.train_r2 = self.evaluate(y, y_pred)
        print(f"  [SVR评估] RMSE={self.train_rmse:.4f} | MAE={self.train_mae:.4f} | R2={self.train_r2:.4f}")
        return y_pred
    
    def predict(self, df_pred):
        X = df_pred[self.features].values
        return self.model.predict(X)
    
    def cross_validate(self, df, n_splits=5):
        print(f"\n[SVR] 开始 {n_splits} 折时序交叉验证...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]
            
            X_train = df_train[self.features].values
            y_train = df_train['B'].values
            X_test = df_test[self.features].values
            y_test = df_test['B'].values
            
            fold_model = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.5))
            ])
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            
            rmse, mae, r2 = self.evaluate(y_test, y_pred)
            fold_results.append({'fold': fold, 'rmse': rmse, 'mae': mae, 'r2': r2})
            print(f"  折 {fold}/{n_splits} | RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")
        
        mean_rmse = np.mean([r['rmse'] for r in fold_results])
        mean_mae = np.mean([r['mae'] for r in fold_results])
        mean_r2 = np.mean([r['r2'] for r in fold_results])
        
        print(f"  [交叉验证均值] RMSE={mean_rmse:.4f} | MAE={mean_mae:.4f} | R2={mean_r2:.4f}\n")
        return fold_results, {'mean_rmse': mean_rmse, 'mean_mae': mean_mae, 'mean_r2': mean_r2}


class RidgeRegressor(ModelBase):
    def __init__(self):
        super().__init__()
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('ridge', Ridge(alpha=1.0))
        ])
        self.features = ['A']
        
    def train(self, df):
        print("\n" + "="*50)
        print("[Ridge回归] 正在训练...")
        print("="*50)
        
        X = df[self.features].values
        y = df['B'].values
        
        print(f"  训练样本数: {len(X)}")
        print(f"  特征数: {len(self.features)}")
        print(f"  多项式阶数: 2")
        print(f"  正则化参数: alpha=1.0")
        
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.train_rmse, self.train_mae, self.train_r2 = self.evaluate(y, y_pred)
        print(f"  [Ridge回归评估] RMSE={self.train_rmse:.4f} | MAE={self.train_mae:.4f} | R2={self.train_r2:.4f}")
        return y_pred
    
    def predict(self, df_pred):
        X = df_pred[self.features].values
        return self.model.predict(X)
    
    def cross_validate(self, df, n_splits=5):
        print(f"\n[Ridge回归] 开始 {n_splits} 折时序交叉验证...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]
            
            X_train = df_train[self.features].values
            y_train = df_train['B'].values
            X_test = df_test[self.features].values
            y_test = df_test['B'].values
            
            fold_model = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('ridge', Ridge(alpha=1.0))
            ])
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            
            rmse, mae, r2 = self.evaluate(y_test, y_pred)
            fold_results.append({'fold': fold, 'rmse': rmse, 'mae': mae, 'r2': r2})
            print(f"  折 {fold}/{n_splits} | RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")
        
        mean_rmse = np.mean([r['rmse'] for r in fold_results])
        mean_mae = np.mean([r['mae'] for r in fold_results])
        mean_r2 = np.mean([r['r2'] for r in fold_results])
        
        print(f"  [交叉验证均值] RMSE={mean_rmse:.4f} | MAE={mean_mae:.4f} | R2={mean_r2:.4f}\n")
        return fold_results, {'mean_rmse': mean_rmse, 'mean_mae': mean_mae, 'mean_r2': mean_r2}


class SVRRegressor(ModelBase):
    def __init__(self):
        super().__init__()
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.5))
        ])
        self.features = ['A']
        
    def train(self, df):
        print("\n" + "="*50)
        print("[SVR] 正在训练...")
        print("="*50)
        
        X = df[self.features]
        y = df['B']
        
        print(f"  训练样本数: {len(X)}")
        print(f"  核函数: RBF")
        print(f"  C=10, epsilon=1.0")
        
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.train_rmse, self.train_mae, self.train_r2 = self.evaluate(y, y_pred)
        print(f"  [SVR评估] RMSE={self.train_rmse:.4f} | MAE={self.train_mae:.4f} | R2={self.train_r2:.4f}")
        return y_pred
    
    def predict(self, df_pred):
        X = df_pred[self.features]
        return self.model.predict(X)
    
    def cross_validate(self, df, n_splits=5):
        print(f"\n[SVR] 开始 {n_splits} 折时序交叉验证...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]
            
            X_train = df_train[self.features]
            y_train = df_train['B']
            X_test = df_test[self.features]
            y_test = df_test['B']
            
            fold_model = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=10, epsilon=1.0))
            ])
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            
            rmse, mae, r2 = self.evaluate(y_test, y_pred)
            fold_results.append({'fold': fold, 'rmse': rmse, 'mae': mae, 'r2': r2})
            print(f"  折 {fold}/{n_splits} | RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")
        
        mean_rmse = np.mean([r['rmse'] for r in fold_results])
        mean_mae = np.mean([r['mae'] for r in fold_results])
        mean_r2 = np.mean([r['r2'] for r in fold_results])
        
        print(f"  [交叉验证均值] RMSE={mean_rmse:.4f} | MAE={mean_mae:.4f} | R2={mean_r2:.4f}\n")
        return fold_results, {'mean_rmse': mean_rmse, 'mean_mae': mean_mae, 'mean_r2': mean_r2}

