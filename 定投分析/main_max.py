import ccxt
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import os

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

        # 转换为Pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # 将时间戳转换为北京时间（UTC+8）
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + timedelta(hours=8)

        # 计算涨跌幅
        df = calculate_price_change(df)

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
        first_close = df['close'].iloc[0]
        last_close = df['close'].iloc[-1]
        final_price_change = ((last_close - first_close) / first_close) * 100

        return {
            'Coin': pair,
            'Max Increase (%)': max_increase,
            'Max Decrease (%)': max_decrease,
            'Final Change (%)': final_price_change,
            'Average Change (%)': average_change
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

    results = []

    # 使用进度条获取数据
    for bi in tqdm(bi_list, desc="获取数据"):
        result = fetch_binance_ohlcv(exchange, f'{bi}/USDT', '1h', 7*24, output_folder)
        if result:
            results.append(result)

    # 保存结果
    result_df = pd.DataFrame(results)
    today = datetime.today().strftime('%m_%d')
    output_file = os.path.join('./', f"summary_{today}.csv")
    result_df.to_csv(output_file, index=False)

    print(f"所有结果已保存至{output_file}")

    # 按涨幅显示前10个币种
    top_10 = result_df.sort_values(by='Final Change (%)', ascending=False).head(10)
    print('今日涨幅前10的币种：')
    print(top_10)

if __name__ == "__main__":
    main()
