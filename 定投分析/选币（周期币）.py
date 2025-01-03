import ccxt
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import numpy as np
from scipy.signal import find_peaks

def fetch_binance_ohlcv(exchange, pair: str, timeframe: str = '15m', limit: int = 1000, output_folder: str = "data"):
    """
    获取指定交易对的历史K线数据，并保存为CSV文件。

    参数:
        exchange: CCXT交易所对象
        pair (str): 交易对，例如'BTC/USDT'
        timeframe (str): 时间间隔，例如'1h'，'15m'
        limit (int): 获取的数据点数量
        output_folder (str): 保存文件的文件夹路径

    返回:
        dict: 涨跌幅统计摘要
    """
    try:
        # 获取K线数据
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)

        # 如果数据量不足，直接跳过
        if len(ohlcv) < limit:
            print(f"{pair} 数据量不足，跳过。")
            return None

        # 转换为Pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # 将时间戳转换为北京时间（UTC+8）
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + timedelta(hours=8)

        # 计算涨跌幅
        df = calculate_price_change(df)

        # 计算主要周期
        dominant_period = calculate_dominant_period(df['close'].dropna())

        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 保存数据到CSV文件
        output_file = os.path.join(output_folder, f"{pair.replace('/', '_')}_{timeframe}.csv")
        df.to_csv(output_file, index=False)

        print(f"成功获取{pair} {timeframe}数据，保存至{output_file}")

        # 计算统计数据
        max_increase = df['price_change'].max()
        max_decrease = df['price_change'].min()
        average_change = df['price_change'].mean()
        volatility = df['price_change'].std()  # 添加波动率计算
        first_close = df['close'].iloc[0]
        last_close = df['close'].iloc[-1]
        final_price_change = ((last_close - first_close) / first_close) * 100

        # 检查价格是否低于10
        below_10 = df['close'].mean() < 10

        return {
            'Coin': pair,
            'Max Increase (%)': max_increase,
            'Max Decrease (%)': max_decrease,
            'Final Change (%)': final_price_change,
            'Average Change (%)': average_change,
            'Volatility (%)': volatility,
            'Dominant Period': dominant_period,
            'Below 10': below_10
        }

    except ccxt.BaseError as e:
        print(f"获取{pair}数据时出错: {e}")
        return None

def calculate_price_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算收盘价的百分比变化。

    参数:
        df (pd.DataFrame): 包含收盘价的DataFrame，需包含'close'列

    返回:
        pd.DataFrame: 添加了'price_change'列的DataFrame
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame必须包含'close'列。")

    # 计算百分比变化
    df['price_change'] = df['close'].pct_change() * 100
    return df

def calculate_dominant_period(series: pd.Series) -> float:
    """
    计算时间序列的主要周期。

    参数:
        series (pd.Series): 时间序列数据，例如收盘价

    返回:
        float: 主要周期（单位为时间间隔数）
    """
    if len(series) < 10:
        return None  # 数据量不足以计算周期

    # 计算一阶差分
    diff = series.diff().dropna()

    # 查找峰值
    peaks, _ = find_peaks(diff)

    if len(peaks) < 2:
        return None  # 如果峰值不足两个，无法计算周期

    # 计算相邻峰值之间的平均距离
    periods = np.diff(peaks)
    return np.mean(periods)

def main():
    """主函数，用于获取和分析数据。"""
    # 配置
    type1 = 'swap'
    quote = 'USDT'
    output_folder = "data_all_1h"

    # 初始化Binance交易所
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'proxies': {
            'http': 'http://127.0.0.1:7890',
            'https': 'http://127.0.0.1:7890'
        }
    })

    # 获取市场数据
    markets = exchange.fetch_markets()
    df = pd.DataFrame(markets)

    # 筛选交易对
    df_spot = df[(df['type'] == type1) & (df['quote'] == quote)]
    bi_list = [item[:-4] for item in df_spot['id'].tolist()]

    print(f"发现{len(bi_list)}个类型为{type1}的交易对。")

    # 检查是否需要重新获取数据
    if os.path.exists("summary.csv"):
        choice = input("发现已有数据文件，是否重新获取数据？(y/n): ").strip().lower()
    else:
        choice = "y"

    results = []

    if choice == "y":
        # 使用进度条获取数据
        for bi in tqdm(bi_list, desc="获取数据"):
            result = fetch_binance_ohlcv(exchange, f'{bi}/USDT', '1h', 7*24, output_folder)
            if result:
                results.append(result)

        # 保存结果
        result_df = pd.DataFrame(results)
        result_df.to_csv("summary.csv", index=False)

        print("所有结果已保存至summary.csv")
    else:
        print("读取现有数据文件...")
        result_df = pd.read_csv("summary.csv")

    # 筛选价格低于10且周期性强的前10个币种
    filtered = result_df[result_df['Below 10'] == True]
    top_10_periodic = filtered.sort_values(by='Dominant Period', ascending=True).head(10)

    print('价格低于10且周期性最强的前10个币种：')
    print(top_10_periodic[['Coin', 'Dominant Period', 'Below 10']])

if __name__ == "__main__":
    main()
