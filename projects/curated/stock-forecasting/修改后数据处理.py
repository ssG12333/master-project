import pandas as pd
import numpy as np
import os
import glob
import talib  # 如果没有安装，请 pip install TA-Lib


def calculate_indicators(df):
    """
    根据方案要求构建 9 类技术指标
    """
    # 确保基础列存在
    required = ['close', 'high', 'low']
    if not all(col in df.columns for col in required):
        return df

    # 1. 趋势类
    df['kama'] = talib.KAMA(df['close'], timeperiod=30)
    df['ema'] = talib.EMA(df['close'], timeperiod=30)
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd

    # 2. 动量类
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['roc'] = talib.ROC(df['close'], timeperiod=10)
    df['cmo'] = talib.CMO(df['close'], timeperiod=14)

    # 3. 波动/超买超卖类
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    df['bb_upper'] = upper
    df['bb_lower'] = lower

    return df


def apply_rolling_normalization(df, window=30):
    """
    时序归一化：针对每一个 T=30 的滑动窗口进行 Z-Score 归一化
    """
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'kama', 'ema', 'macd', 'rsi', 'roc', 'cmo', 'atr', 'cci', 'bb_upper', 'bb_lower'
    ]

    cols_to_norm = [c for c in feature_cols if c in df.columns]

    for col in cols_to_norm:
        rolling_mean = df[col].rolling(window=window).mean()
        rolling_std = df[col].rolling(window=window).std()
        df[f'{col}_norm'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

    return df


def process_single_stock(file_path, trading_days):
    """
    处理单只股票：清洗、对齐、生成特征、时序归一化与标签
    """
    stock_code = os.path.basename(file_path).split('_')[0]

    # 兼容性读取
    df = None
    for enc in ['gb18030', 'utf-8', 'gbk']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            break
        except Exception:
            continue

    if df is None:
        return None

    # 清洗列名
    df.columns = [str(c).strip().replace(' ', '') for c in df.columns]

    # 1. 列名标准化
    column_mapping = {
        "日期": "date", "时间": "date",
        "开盘价": "open", "开盘": "open",
        "收盘价": "close", "收盘": "close",
        "最高价": "high", "最高": "high",
        "最低价": "low", "最低": "low",
        "成交量": "volume", "成交量(手)": "volume", "成交量(股)": "volume"
    }
    df.rename(columns=column_mapping, inplace=True)

    # 模糊匹配 volume
    if 'volume' not in df.columns:
        for c in df.columns:
            if '量' in c:
                df.rename(columns={c: 'volume'}, inplace=True)
                break

    # 检查核心列
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in required_cols):
        return None

    # 2. 时间处理
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df = df.sort_values('date').drop_duplicates('date')

    # 3. 对齐交易日历 (解决时间对齐问题)
    df = df.set_index('date').reindex(trading_days)
    df.index.name = 'date'  # 强制命名索引，防止 reset_index 后列名变为 'index'

    # 4. 缺失值处理 (停牌处理)
    df['close'] = df['close'].ffill()
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df['close'])
    df['low'] = df['low'].fillna(df['close'])
    df['volume'] = df['volume'].fillna(0)

    # 5. 计算标签
    df['target_return'] = df['close'].shift(-1) / df['close'] - 1

    # 6. 构建特征与归一化
    df = calculate_indicators(df)
    df = apply_rolling_normalization(df, window=30)

    # 7. 过滤并返回
    df = df.dropna(subset=['kama_norm', 'target_return']).reset_index()
    df['code'] = stock_code
    return df


def main():
    input_dir = 'data'
    output_dir = 'data_processed'
    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not files:
        print("未找到原始数据文件")
        return

    print("正在构建全局交易日历...")
    all_dates = []
    # 采样前50个文件来确定交易日历范围
    for f in files[:50]:
        try:
            for enc in ['gb18030', 'utf-8', 'gbk']:
                try:
                    tmp = pd.read_csv(f, usecols=[0], encoding=enc)
                    all_dates.extend(pd.to_datetime(tmp.iloc[:, 0], errors='coerce').dropna().tolist())
                    break
                except:
                    continue
        except:
            continue

    if not all_dates:
        print("无法提取有效日期")
        return

    # 确定统一的交易日历，并设定截止日期
    trading_days = pd.DatetimeIndex(sorted(list(set(all_dates))))
    trading_days = trading_days[trading_days <= '2025-06-30']
    trading_days.name = 'date'

    print(f"开始处理 {len(files)} 只股票，采用逐个文件保存方式...")

    success_count = 0
    for f in files:
        try:
            processed_df = process_single_stock(f, trading_days)
            if processed_df is not None and len(processed_df) > 0:
                # 格式化日期列为字符串，方便后续读取
                processed_df['date'] = processed_df['date'].dt.strftime('%Y-%m-%d')

                # 按照你原来的方式命名并保存
                stock_code = processed_df['code'].iloc[0]
                output_path = os.path.join(output_dir, f"{stock_code}_processed.csv")
                processed_df.to_csv(output_path, index=False, encoding='utf-8')
                success_count += 1
        except Exception as e:
            print(f"处理失败 {os.path.basename(f)}: {e}")

    print(f"完成! 成功处理并对齐了 {success_count} 个文件。保存目录: {output_dir}")


if __name__ == "__main__":
    main()