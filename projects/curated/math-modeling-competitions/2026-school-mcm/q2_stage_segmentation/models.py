import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

class BaselinePELT:
    def __init__(self, penalty=10):
        self.penalty = penalty
        
    def predict(self, df):
        dt = 1.0/6.0
        n = len(df)
        
        vel = np.gradient(df['Disp_Smooth'].values) / dt
        vel_smooth = gaussian_filter1d(vel, sigma=100)
        vel_smooth = np.maximum(vel_smooth, 1e-8)
        
        vel_raw = np.gradient(df['Disp_Smooth'].values) / dt
        vel_s = gaussian_filter1d(vel_raw, sigma=30)
        vel_s = np.maximum(vel_s, 1e-8)
        
        threshold = 0.6
        sustained = np.where(vel_s > threshold)[0]
        
        cp1 = None
        if len(sustained) > 0:
            window = 300
            for i in range(len(sustained) - window):
                if sustained[i + window] - sustained[i] < window * 1.5:
                    cp1 = sustained[i]
                    break
        
        if cp1 is None:
            fallback = np.where(vel_s > 0.5)[0]
            cp1 = fallback[0] if len(fallback) > 0 else 7800
        
        vel_dd = np.gradient(np.gradient(vel_smooth))
        vel_dd_smooth = gaussian_filter1d(vel_dd, sigma=200)
        
        peaks_vel = argrelextrema(vel_dd_smooth, np.greater, order=300)[0]
        
        if len(peaks_vel) > 0:
            cp2 = peaks_vel[-1]
        else:
            v0 = np.mean(vel_smooth[:200])
            vel_ratio = vel_smooth / v0
            ratio_100 = np.where(vel_ratio > 100)[0]
            cp2 = ratio_100[0] if len(ratio_100) > 0 else n * 4 // 5
        
        cp1 = max(7000, min(cp1, 8500))
        cp2 = max(cp1 + 500, min(cp2, n - 50))
        
        return cp1, cp2


def linear_model(t, k, c):
    return k * t + c

def exponential_model(t, A, B, C):
    t_clipped = np.clip(t, -np.inf, 50)
    return A * np.exp(B * t_clipped) + C

def power_model(t, A, B, C):
    t_positive = np.maximum(t, 0.1)
    return A * t_positive**B + C

def voight_model(t, tf, lam, C):
    t_safe = np.clip(t, -np.inf, tf - 0.01)
    return -np.log(np.maximum(lam * (tf - t_safe), 1e-10)) / lam + C


class Q2PhysicsModeler:
    def __init__(self):
        self.results = {}
        
    def fit_all_stages(self, df, cp1, cp2):
        dt = 1.0 / 6.0
        t_all = np.arange(len(df)) * dt
        x_all = df['Disp_Smooth'].values
        v_all = df['Velocity'].values
        
        results = {}
        
        t1, x1 = t_all[:cp1], x_all[:cp1]
        t2, x2 = t_all[cp1:cp2], x_all[cp1:cp2]
        t3, x3 = t_all[cp2:], x_all[cp2:]
        v3 = v_all[cp2:]
        
        results['stage1'] = self._fit_stage1(t1, x1)
        results['stage2'] = self._fit_stage2(t2 - t2[0], x2)
        results['stage3'] = self._fit_stage3(t3 - t3[0], x3, v3)
        
        results['cp1'] = cp1
        results['cp2'] = cp2
        results['t_cp1_h'] = cp1 * dt
        results['t_cp2_h'] = cp2 * dt
        
        avg_v1 = (x_all[cp1] - x_all[0]) / (cp1 * dt) if cp1 > 0 else 0
        avg_v2 = (x_all[cp2] - x_all[cp1]) / ((cp2 - cp1) * dt) if cp2 > cp1 else 0
        avg_v3 = (x_all[-1] - x_all[cp2]) / ((len(df) - cp2) * dt) if len(df) > cp2 else 0
        results['avg_velocities'] = (avg_v1, avg_v2, avg_v3)
        
        self.results = results
        return results
        
    def _fit_stage1(self, t, x):
        try:
            popt, pcov = curve_fit(linear_model, t, x, p0=[0.01, x[0]], maxfev=5000)
            pred = linear_model(t, *popt)
            rmse = np.sqrt(np.mean((x - pred)**2))
            r2 = 1 - np.sum((x - pred)**2) / (np.sum((x - np.mean(x))**2) + 1e-10)
            return {
                'model': 'Linear: x(t) = k*t + c',
                'params': {'k': popt[0], 'c': popt[1]},
                'rmse': rmse, 'r2': r2,
                'params_str': f'k={popt[0]:.6f}, c={popt[1]:.4f}'
            }
        except:
            return {'model': 'Linear', 'params': {}, 'rmse': 1e6, 'r2': 0, 'params_str': 'N/A'}
            
    def _fit_stage2(self, t, x):
        best_result = None
        best_rmse = np.inf
        try:
            popt, _ = curve_fit(exponential_model, t, x, p0=[1.0, 0.001, x[0]], maxfev=5000)
            pred = exponential_model(t, *popt)
            rmse = np.sqrt(np.mean((x - pred)**2))
            r2 = 1 - np.sum((x - pred)**2) / (np.sum((x - np.mean(x))**2) + 1e-10)
            if rmse < best_rmse:
                best_rmse = rmse
                best_result = {
                    'model': 'Exponential: x(t) = A*exp(B*t) + C',
                    'params': {'A': popt[0], 'B': popt[1], 'C': popt[2]},
                    'rmse': rmse, 'r2': r2,
                    'params_str': f'A={popt[0]:.4f}, B={popt[1]:.6f}, C={popt[2]:.4f}'
                }
        except:
            pass
        try:
            popt2, _ = curve_fit(power_model, t, x, p0=[1.0, 2.0, x[0]], maxfev=5000)
            pred2 = power_model(t, *popt2)
            rmse2 = np.sqrt(np.mean((x - pred2)**2))
            r2_2 = 1 - np.sum((x - pred2)**2) / (np.sum((x - np.mean(x))**2) + 1e-10)
            if rmse2 < best_rmse:
                best_result = {
                    'model': 'Power: x(t) = A*t^B + C',
                    'params': {'A': popt2[0], 'B': popt2[1], 'C': popt2[2]},
                    'rmse': rmse2, 'r2': r2_2,
                    'params_str': f'A={popt2[0]:.4f}, B={popt2[1]:.4f}, C={popt2[2]:.4f}'
                }
        except:
            pass
        if best_result is None:
            best_result = {'model': 'Power', 'params': {}, 'rmse': 1e6, 'r2': 0, 'params_str': 'N/A'}
        return best_result
        
    def _fit_stage3(self, t, x, v):
        best_result = None
        best_rmse = np.inf
        try:
            t_max = t[-1]
            tf_guess = t_max * 1.5
            popt, _ = curve_fit(voight_model, t, x, p0=[tf_guess, 0.1, x[0]], maxfev=5000, bounds=([t_max*1.01, 1e-6, -np.inf], [t_max*10, 10, np.inf]))
            pred = voight_model(t, *popt)
            rmse = np.sqrt(np.mean((x - pred)**2))
            r2 = 1 - np.sum((x - pred)**2) / (np.sum((x - np.mean(x))**2) + 1e-10)
            if rmse < best_rmse:
                best_rmse = rmse
                best_result = {
                    'model': 'Voight/Saito: x(t) = -ln(λ*(tf-t))/λ + C',
                    'params': {'tf': popt[0], 'lambda': popt[1], 'C': popt[2]},
                    'rmse': rmse, 'r2': r2,
                    'params_str': f'tf={popt[0]:.2f}h, λ={popt[1]:.6f}, C={popt[2]:.4f}',
                    'tf_hours': popt[0]
                }
        except:
            pass
        try:
            popt2, _ = curve_fit(exponential_model, t, x, p0=[1.0, 0.01, x[0]], maxfev=5000)
            pred2 = exponential_model(t, *popt2)
            rmse2 = np.sqrt(np.mean((x - pred2)**2))
            r2_2 = 1 - np.sum((x - pred2)**2) / (np.sum((x - np.mean(x))**2) + 1e-10)
            if rmse2 < best_rmse:
                best_result = {
                    'model': 'Exponential: x(t) = A*exp(B*t) + C',
                    'params': {'A': popt2[0], 'B': popt2[1], 'C': popt2[2]},
                    'rmse': rmse2, 'r2': r2_2,
                    'params_str': f'A={popt2[0]:.4f}, B={popt2[1]:.6f}, C={popt2[2]:.4f}'
                }
        except:
            pass
        if best_result is None:
            best_result = {'model': 'Voight', 'params': {}, 'rmse': 1e6, 'r2': 0, 'params_str': 'N/A'}
        return best_result
    
    def print_summary(self):
        r = self.results
        print("\n" + "="*70)
        print("  三阶段数学模型拟合结果")
        print("="*70)
        
        print(f"\n转换节点:")
        print(f"  tc1 = {r['cp1']} ({r['t_cp1_h']:.1f}h)")
        print(f"  tc2 = {r['cp2']} ({r['t_cp2_h']:.1f}h)")
        
        print(f"\n各阶段平均速度:")
        v1, v2, v3 = r['avg_velocities']
        print(f"  阶段1 (缓慢匀速): {v1:.4f} mm/h")
        print(f"  阶段2 (加速形变): {v2:.4f} mm/h")
        print(f"  阶段3 (快速失稳): {v3:.4f} mm/h")
        
        for stage_name, key in [("阶段1 (缓慢匀速期)", 'stage1'), 
                                ("阶段2 (加速形变期)", 'stage2'), 
                                ("阶段3 (快速失稳期)", 'stage3')]:
            s = r[key]
            print(f"\n{stage_name}:")
            print(f"  模型: {s['model']}")
            print(f"  参数: {s['params_str']}")
            print(f"  RMSE: {s['rmse']:.4f}")
            print(f"  R²:   {s['r2']:.6f}")
            
        if 'tf_hours' in r['stage3']['params']:
            tf_relative = r['stage3']['params']['tf']
            tf_absolute = r['t_cp2_h'] + tf_relative
            print(f"\n预警分析 (Saito/Voight):")
            print(f"  预测失稳时刻: tc2 + {tf_relative:.1f}h = {tf_absolute:.1f}h (从观测起点)")
            
        print("="*70)


class NoiseDiscriminator:
    @staticmethod
    def detect_noise_jumps(df, threshold_sigma=5.0):
        disp = df['Displacement'].values
        velocity = np.diff(disp)
        vel_mean = np.mean(velocity[:100])
        vel_std = np.std(velocity[:100])
        
        noise_indices = []
        for i in range(1, len(velocity)):
            if i > 0 and i < len(velocity) - 1:
                vel_before = velocity[max(0, i-5):i].mean()
                vel_after = velocity[i+1:min(len(velocity), i+6)].mean()
                vel_current = velocity[i]
                
                if abs(vel_current - vel_mean) > threshold_sigma * vel_std:
                    if abs(vel_before - vel_mean) < 2 * vel_std and abs(vel_after - vel_mean) < 2 * vel_std:
                        noise_indices.append(i)
        return noise_indices
    
    @staticmethod
    def is_real_transition(df, cp_index, window=200):
        velocity = df['Velocity'].values
        n = len(velocity)
        
        vel_before = velocity[max(0, cp_index-window):cp_index]
        vel_after = velocity[cp_index:min(n, cp_index+window)]
        
        mean_before = np.mean(vel_before)
        mean_after = np.mean(vel_after)
        
        trend_before = np.polyfit(np.arange(len(vel_before)), vel_before, 1)[0]
        trend_after = np.polyfit(np.arange(len(vel_after)), vel_after, 1)[0]
        
        is_persistent = mean_after > mean_before * 1.5
        is_monotonic = trend_after > 0
        
        return is_persistent and is_monotonic
