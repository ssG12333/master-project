import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from q2_data_processor import Q2DataProcessor
from q2_models import BaselinePELT, Q2PhysicsModeler, NoiseDiscriminator
from q2_visualization import Q2Visualizer

def main():
    print("="*70)
    print("  边坡预警系统 - 问题2：三段式识别与物理建模")
    print("="*70)
    
    filepath = os.path.join(os.path.dirname(__file__), "..", "Attachment 2：Displacement time series data – Question 2 copy.xlsx")
    filepath = os.path.abspath(filepath)
    
    print("\n[步骤 1] 正在加载并清洗数据 (Hampel + SG 滤波)...")
    processor = Q2DataProcessor(filepath)
    try:
        df = processor.load_and_clean()
        df = processor.extract_features(df)
        print(f"  数据处理完毕，总时间步: {len(df)}")
    except FileNotFoundError:
        print(f"  找不到文件: {filepath}")
        return
    
    noise_indices = NoiseDiscriminator.detect_noise_jumps(df)
    if noise_indices:
        print(f"  检测到 {len(noise_indices)} 个噪声跳变点，已通过 Hampel 滤波剔除")
    
    print("\n[步骤 2] 执行转换节点识别 (双准则: 速度跃升 + 速度二阶导)...")
    
    pelt = BaselinePELT()
    cp1, cp2 = pelt.predict(df)
    
    tc1 = NoiseDiscriminator.is_real_transition(df, cp1)
    tc2 = NoiseDiscriminator.is_real_transition(df, cp2)
    
    print(f"  转换节点 tc1 = {cp1} ({cp1/6:.1f}h), 真实转换: {'是' if tc1 else '否'}")
    print(f"  转换节点 tc2 = {cp2} ({cp2/6:.1f}h), 真实转换: {'是' if tc2 else '否'}")
    
    print("\n[步骤 3] 三阶段数学模型拟合与检验...")
    modeler = Q2PhysicsModeler()
    model_results = modeler.fit_all_stages(df, cp1, cp2)
    modeler.print_summary()
    
    print("\n[步骤 4] 渲染学术图表...")
    vis = Q2Visualizer()
    vis.plot_raw_data(df)
    vis.plot_preprocessing_comparison(df)
    vis.plot_noise_vs_real_transition(df)
    vis.plot_tc1_identification(df, cp1)
    vis.plot_tc2_identification(df, cp1, cp2)
    vis.plot_stage1_model(df, model_results)
    vis.plot_stage2_model(df, model_results)
    vis.plot_stage3_model(df, model_results)
    vis.plot_stage_colored_displacement(df, cp1, cp2)
    vis.plot_kde_comparison(df, cp1, cp2)
    vis.plot_comprehensive_judgment(df, cp1, cp2)
    vis.plot_all_stages_combined(df, model_results)
    
    print("\n" + "="*70)
    print("  第二问执行完毕！共生成10张学术图表")
    print("="*70)

if __name__ == "__main__":
    main()
