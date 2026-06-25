import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def run_backtest(prediction_file="gru_predictions.csv", top_k=10):
    """
    运行 Top-K 交易模拟 (Top-10 策略)

    参数:
    prediction_file: 包含 'date', 'code', 'actual_return', 'pred_return' 的 CSV 文件
    top_k: 每日买入预测收益率最高的股票数量
    """
    if not os.path.exists(prediction_file):
        print(f"错误: 找不到预测结果文件 {prediction_file}")
        return

    # 1. 加载数据
    df = pd.read_csv(prediction_file)
    df['date'] = pd.to_datetime(df['date'])

    # 2. 策略模拟：每日选择 Top-K 股票进行等权投资
    # 这里的逻辑是：在 T 日收盘后获得 T+1 日的预测值，并在 T+1 日开盘买入，收盘计算收益
    print(f"正在模拟 Top-{top_k} 选股策略...")

    # 按日期分组计算每日收益
    def get_daily_returns(group):
        # 排除预测值为 NaN 的数据
        group = group.dropna(subset=['pred_return', 'actual_return'])
        if len(group) < top_k:
            return pd.Series({'strategy_return': 0, 'benchmark_return': 0})

        # 按预测收益率降序排列，取前 K 名
        top_stocks = group.sort_values(by='pred_return', ascending=False).head(top_k)

        # 计算这 K 只股票的真实收益率均值 (等权持仓)
        strategy_ret = top_stocks['actual_return'].mean()

        # 基准收益率：全市场等权收益
        benchmark_ret = group['actual_return'].mean()

        return pd.Series({
            'strategy_return': strategy_ret,
            'benchmark_return': benchmark_ret
        })

    # 计算每日策略收益
    daily_results = df.groupby('date').apply(get_daily_returns)

    # 3. 计算累计收益曲线
    daily_results['cum_strategy'] = (1 + daily_results['strategy_return']).cumprod()
    daily_results['cum_benchmark'] = (1 + daily_results['benchmark_return']).cumprod()

    # 4. 核心评估指标计算
    # 计算回测天数和年化系数 (假设一年 252 个交易日)
    days = len(daily_results)
    annual_factor = 252 / days if days > 0 else 0

    # A. 年化收益率 (Annualized Return)
    total_return = daily_results['cum_strategy'].iloc[-1] - 1
    annual_return = (1 + total_return) ** annual_factor - 1

    # B. 夏普比率 (Sharpe Ratio)
    # 假设无风险利率为 2%
    risk_free_rate = 0.02
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_return = daily_results['strategy_return'] - daily_rf
    # 年化夏普 = (日均超额收益 / 收益标准差) * sqrt(252)
    if daily_results['strategy_return'].std() != 0:
        sharpe_ratio = (daily_results['strategy_return'].mean() / daily_results['strategy_return'].std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # C. 最大回撤 (Maximum Drawdown)
    cumulative_max = daily_results['cum_strategy'].cummax()
    drawdown = (daily_results['cum_strategy'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    # 5. 输出回测报告
    start_date = daily_results.index.min().date()
    end_date = daily_results.index.max().date()

    print("\n" + "=" * 40)
    print("         量化策略回测报告 (Top-K)")
    print("=" * 40)
    print(f"回测周期 (时间段): {start_date} 至 {end_date}")
    print(f"有效交易天数: {days} 天")
    print(f"选股策略: 每日买入预测收益率前 {top_k} 名")
    print("-" * 40)
    print(f"1. 年化收益率: {annual_return:.2%}")
    print(f"2. 夏普比率:   {sharpe_ratio:.2f}")
    print(f"3. 最大回撤:   {max_drawdown:.2%}")
    print("-" * 40)
    print(f"累计策略收益: {total_return:.2%}")
    print(f"累计基准收益: {daily_results['cum_benchmark'].iloc[-1] - 1:.2%}")
    print("=" * 40)

    # 6. 可视化对比图
    plt.figure(figsize=(12, 6))
    plt.plot(daily_results.index, daily_results['cum_strategy'], label=f'Strategy (Top-{top_k})', color='red', linewidth=2)
    plt.plot(daily_results.index, daily_results['cum_benchmark'], label='Market (Equal Weight)', color='gray', linestyle='--', alpha=0.7)
    plt.title(f'Cumulative Returns: Top-{top_k} Strategy vs Market')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 保存结果图片
    plt.savefig('strategy_backtest.png', dpi=300)
    print(f"\n可视化图表已保存为: strategy_backtest.png")
    plt.show()


if __name__ == "__main__":
    # 按照要求执行 Top-10 策略回测
    run_backtest(prediction_file="gru_predictions.csv", top_k=10)