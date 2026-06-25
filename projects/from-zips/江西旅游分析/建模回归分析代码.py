
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels import PanelOLS
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
print("可用样式列表:", plt.style.available)
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')  # 设置专业图表样式
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

        # 数据清洗
        df = df[~df['地区'].str.contains('江西|全省', na=False)]  # 排除汇总数据
        df = df.dropna(subset=['旅游竞争力得分'])  # 必须保留被解释变量

        # 类型转换与异常值处理
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
        df[num_cols] = df[num_cols].clip(lower=df[num_cols].quantile(0.01),
                                         upper=df[num_cols].quantile(0.99),
                                         axis=1)

        # 设置面板数据结构
        df = df.set_index(['地区', '年份']).sort_index()
        return df
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit()


# ==================== 描述性分析 ====================
def descriptive_analysis(df):
    """生成增强型描述统计"""
    desc = df.describe().T
    desc['缺失值'] = df.isnull().sum()
    desc['变异系数'] = df.std() / df.mean()
    return desc.round(3)


# ==================== 计量模型 ====================
def panel_regression(df):
    """稳健的双向固定效应模型，仅使用指定的解释变量"""
    y_var = '旅游竞争力得分'
    X_vars = ['5A级景区数量', '人均GDP', '住宿餐饮业从业人员数(万人)', '星级酒店数量', '国内旅游人数']

    # 检查所有指定变量是否在数据集中存在
    missing_vars = [var for var in X_vars if var not in df.columns]
    if missing_vars:
        print(f"以下变量在数据集中不存在: {missing_vars}")
        exit()

    # 模型配置
    model = PanelOLS(
        dependent=df[y_var],
        exog=df[X_vars],
        entity_effects=True,  # 个体固定效应
        time_effects=True,   # 时间固定效应
        drop_absorbed=True   # 移除被吸收的变量
    )

    # 模型估计
    try:
        results = model.fit(cov_type='clustered', cluster_entity=True)
        return results, X_vars
    except Exception as e:
        print(f"模型估计失败: {str(e)}")
        exit()


# ==================== 模型诊断 ====================
def model_diagnostics(df, results, valid_vars):
    """综合模型检验"""
    # 多重共线性检验
    X = sm.add_constant(df[valid_vars])
    vif = pd.DataFrame()
    vif["变量"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # 异方差检验
    bp_test = het_breuschpagan(results.resids, X)
    bp_result = pd.DataFrame([bp_test],
                             columns=['LM统计量', 'P值', 'F统计量', 'F P值'],
                             index=['Breusch-Pagan检验'])

    return vif.round(2), bp_result.round(4)


# ==================== 结果可视化 ====================
def visualize_results(results):
    """专业级结果可视化"""
    coef = results.params.copy()

    # 过滤固定效应标识
    coef = coef[~coef.index.str.contains('effect|const', case=False)]

    # 创建图表
    plt.figure(figsize=(12, 8))
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in coef]
    ax = coef.plot(kind='barh', color=colors, edgecolor='k', alpha=0.8)

    # 添加显著性标记
    pvals = results.pvalues[coef.index]
    for i, (name, val) in enumerate(coef.items()):
        sig = ''
        if pvals[name] < 0.01:
            sig = '***'
        elif pvals[name] < 0.05:
            sig = '**'
        elif pvals[name] < 0.1:
            sig = '*'
        ax.text(val / 2 if val > 0 else val * 1.2, i, sig, ha='center', va='center', color='white')

    # 图表装饰
    plt.title('旅游竞争力驱动因素分析结果\n（颜色表示作用方向，*表示显著性水平）', fontsize=14)
    plt.xlabel('回归系数', fontsize=12)
    plt.ylabel('解释变量', fontsize=12)
    plt.grid(axis='x', linestyle='--')
    plt.savefig('回归结果可视化.png')


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 数据准备
    df = load_data()
    print("数据加载完成，包含{}个地区×{}个年份".format(
        df.index.get_level_values(0).nunique(),
        df.index.get_level_values(1).nunique()
    ))

    # 描述性分析
    desc = descriptive_analysis(df)
    with pd.ExcelWriter('分析结果.xlsx', engine='openpyxl', mode='w') as writer:
        desc.to_excel(writer, sheet_name='描述统计')

    # 相关性分析
    corr = df.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
                annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
    plt.title('变量相关系数矩阵', fontsize=14)
    plt.savefig('相关系数矩阵.png')

    # 计量分析
    results, valid_vars = panel_regression(df)
    print("\n回归模型估计成功！\nR² = {:.3f}".format(results.rsquared))

    # 模型诊断
    vif, bp = model_diagnostics(df, results, valid_vars)
    with pd.ExcelWriter('分析结果.xlsx', engine='openpyxl', mode='a') as writer:
        vif.to_excel(writer, sheet_name='多重共线性检验', index=False)
        bp.to_excel(writer, sheet_name='异方差检验')

    # 结果保存
    with open('回归结果.txt', 'w', encoding='utf-8') as f:
        f.write(str(results))

    # 可视化
    visualize_results(results)

    print("""
    分析完成！生成文件清单：
    1. 分析结果.xlsx —— 含描述统计、检验结果
    2. 回归结果.txt —— 详细回归输出
    3. 相关系数矩阵.png —— 变量相关关系图
    4. 回归结果可视化.png —— 核心结果图示
    """)