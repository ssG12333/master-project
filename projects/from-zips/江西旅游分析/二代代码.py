import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels import PooledOLS, PanelOLS, RandomEffects, compare
from scipy.stats import f, chi2
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
from linearmodels.panel.tests import BreuschPaganTest

warnings.filterwarnings('ignore')

plt.style.use('seaborn')  # 设置图表样式
plt.rcParams.update({
    'font.sans-serif': 'SimHei',
    'axes.unicode_minus': False,
    'savefig.dpi': 300,
    'figure.autolayout': True
})


# ==================== 数据准备 ====================
def load_data():
    """智能数据加载与清洗"""
    try:
        df = pd.read_excel('总表.xlsx', sheet_name='Sheet1', engine='openpyxl')
        # 列名标准化
        df = df.rename(columns={
            '旅游竞争力得分': 'Tc',
            '5A景区数量': 'Tce',
            '人均GDP': 'Se',
            '旅游从业人数': 'Hr',
            '星级酒店数量': 'Ti',
            '国内旅游人数': 'Tcm'
        })

        # 数据清洗
        df = df[~df['地区'].str.contains('江西|全省', na=False)]
        df = df.dropna(subset=['Tc'])

        # 设置面板数据结构
        df = df.set_index(['地区', '年份']).sort_index()
        return df
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit()


# ==================== 计量模型 ====================
def panel_analysis(df):
    """执行三种模型估计与检验"""
    y_var = 'Tc'
    X_vars = ['Tce', 'Se', 'Hr', 'Ti', 'Tcm']

    # 准备数据
    data = df[[y_var] + X_vars].dropna()
    y = data[y_var]
    X = data[X_vars]

    # 模型估计
    # 混合模型（添加截距项）
    X_pool = sm.add_constant(X)
    pool_model = PooledOLS(y, X_pool)
    pool_res = pool_model.fit(cov_type='clustered', cluster_entity=True)

    # 固定效应模型（不添加截距项）
    fe_model = PanelOLS(y, X, entity_effects=True)
    fe_res = fe_model.fit(cov_type='clustered', cluster_entity=True)

    # 随机效应模型（添加截距项）
    re_model = RandomEffects(y, X_pool)
    re_res = re_model.fit(cov_type='clustered', cluster_entity=True)

    # ========== 模型检验 ==========
    # F检验（混合 vs 固定）
    f_test = fe_res.f_pooled

    # BP检验（混合 vs 随机）

    bp_stat, bp_pval, _, _ = BreuschPaganTest.run(pool_res)

    # Hausman检验（固定 vs 随机）
    try:
        hausman = fe_res.compare(re_res)
        hausman_stat = hausman.stat.ix['Hausman']
        hausman_pval = hausman.pval.ix['Hausman']
    except:
        hausman_stat, hausman_pval = np.nan, np.nan

    return {
        'pool': pool_res,
        'fe': fe_res,
        're': re_res,
        'tests': {
            'F检验': (f_test.stat, f_test.pval),
            'BP检验': (bp_stat, bp_pval),
            'Hausman检验': (hausman_stat, hausman_pval)
        }
    }


# ==================== 结果可视化 ====================
def visualize_comparison(results):
    """模型系数对比可视化"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # 提取系数
    fe_coef = results['fe'].params
    re_coef = results['re'].params[fe_coef.index]  # 对齐变量
    pool_coef = results['pool'].params[fe_coef.index]

    # 创建DataFrame
    coef_df = pd.DataFrame({
        '固定效应': fe_coef,
        '随机效应': re_coef,
        '混合模型': pool_coef
    })

    # 绘制系数对比
    coef_df.plot(kind='barh', ax=ax, alpha=0.8)
    ax.set_title('模型系数对比（标准化处理后）', fontsize=14)
    ax.set_xlabel('回归系数', fontsize=12)
    ax.axvline(0, color='gray', linestyle='--')
    plt.savefig('模型对比.png', bbox_inches='tight')


# ==================== 主程序 ====================
if __name__ == "__main__":
    df = load_data()
    print("数据维度:", df.shape)

    # 执行分析
    results = panel_analysis(df)

    # 输出检验结果
    tests = results['tests']
    print("\n=== 模型选择检验 ===")
    print(f"F检验统计量: {tests['F检验'][0]:.2f} (p={tests['F检验'][1]:.3f})")
    print(f"BP检验统计量: {tests['BP检验'][0]:.2f} (p={tests['BP检验'][1]:.3f})")
    print(f"Hausman检验统计量: {tests['Hausman检验'][0]:.2f} (p={tests['Hausman检验'][1]:.3f})")

    # 模型选择逻辑
    if tests['Hausman检验'][1] < 0.05:
        final_model = results['fe']
        print("\n根据Hausman检验，选择固定效应模型")
    elif tests['BP检验'][1] < 0.05:
        final_model = results['re']
        print("\n根据BP检验，选择随机效应模型")
    else:
        final_model = results['pool']
        print("\n根据检验结果，选择混合模型")

    # 保存最终结果
    with open('final_model.txt', 'w') as f:
        f.write(str(final_model))

    # 可视化对比
    visualize_comparison(results)

    print("""
    分析完成！生成文件：
    1. final_model.txt - 最终模型结果
    2. 模型对比.png - 系数对比图
    """)