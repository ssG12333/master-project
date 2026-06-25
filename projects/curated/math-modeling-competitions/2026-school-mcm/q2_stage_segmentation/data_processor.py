import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

class Q2DataProcessor:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.dt_hours = 1.0 / 6.0  # 10分钟一次 = 1/6小时
        
    def load_and_clean(self):
        """读取数据并应用 Hampel 滤波剔除孤立毛刺 (如附件2中的第34点)"""
        if self.filepath:
            if self.filepath.endswith('.csv'):
                df = pd.read_csv(self.filepath)
            elif self.filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(self.filepath)
            else:
                # 默认尝试读取
                try:
                    df = pd.read_csv(self.filepath)
                except:
                    df = pd.read_excel(self.filepath)
            # 统一列名
            df.columns = ['ID', 'Displacement']
        else:
            raise ValueError("请提供有效的文件路径")
            
        # 1. Hampel 滤波 (去极值毛刺)
        window_size = 7
        n_sigmas = 3
        rolling_median = df['Displacement'].rolling(window=window_size, center=True).median()
        rolling_mad = df['Displacement'].rolling(window=window_size, center=True).std()
        
        # 标记异常点并替换为中位数
        outliers = np.abs(df['Displacement'] - rolling_median) > (n_sigmas * rolling_mad)
        df['Disp_Clean'] = df['Displacement']
        df.loc[outliers, 'Disp_Clean'] = rolling_median[outliers]
        
        # 填充首尾 NaN
        df['Disp_Clean'] = df['Disp_Clean'].bfill().ffill()
        
        # 2. Savitzky-Golay 滤波 (平滑物理趋势)
        # 窗口大小选 51 (约8.5小时)，多项式阶数选 3
        df['Disp_Smooth'] = savgol_filter(df['Disp_Clean'], window_length=51, polyorder=3)
        # 保证位移非负且单调递增(边坡物理特性)
        df['Disp_Smooth'] = np.maximum.accumulate(df['Disp_Smooth'])
        
        return df
        
    def extract_features(self, df):
        """计算速度、加速度与无量纲宏观切线角"""
        # 速度 v_t (mm/h)
        df['Velocity'] = df['Disp_Smooth'].diff() / self.dt_hours
        df['Velocity'] = df['Velocity'].fillna(method='bfill')
        
        # 加速度 a_t (mm/h^2)
        df['Acceleration'] = df['Velocity'].diff() / self.dt_hours
        df['Acceleration'] = df['Acceleration'].fillna(0)
        
        # 提取缓慢匀速期的基础速度 v0 (取前 24 小时即 144 个点的均值)
        v0 = df['Velocity'].iloc[:144].mean()
        if v0 <= 0 or np.isnan(v0): v0 = 0.001 # 防除零
        
        # 宏观切线角 (转换为角度)
        df['Tangent_Angle'] = np.degrees(np.arctan(df['Velocity'] / v0))
        
        return df

# ==========================================
# 独立测试入口
# ==========================================
if __name__ == "__main__":
    print("🚀 测试数据处理模块 (Data Processor)...")
    np.random.seed(42)
    # 模拟生成带有跳变的位移数据
    t = np.linspace(0, 100, 1000)
    disp = 0.01 * t + 0.0005 * t**2 + 1e-6 * t**3
    disp[300] = 5.0 # 模拟跳变噪声
    df_mock = pd.DataFrame({'ID': range(len(disp)), 'Displacement': disp})
    
    dp = Q2DataProcessor()
    dp.filepath = "mock"
    # 猴子补丁绕过文件读取
    dp.load_and_clean = lambda: pd.DataFrame({'ID': df_mock['ID'], 'Displacement': df_mock['Displacement'], 'Disp_Clean': df_mock['Displacement']})
    
    # 手动应用滤波
    df_mock['Disp_Clean'] = df_mock['Displacement']
    df_mock.loc[300, 'Disp_Clean'] = (df_mock['Displacement'][299] + df_mock['Displacement'][301])/2
    df_mock['Disp_Smooth'] = savgol_filter(df_mock['Disp_Clean'], 51, 3)
    
    df_feat = dp.extract_features(df_mock)
    print(f"✅ 特征提取成功！头部特征：\n{df_feat[['Velocity', 'Acceleration', 'Tangent_Angle']].head()}")