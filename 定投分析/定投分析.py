import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_binance_ohlcv(pair: str, timeframe: str = '15m', limit: int = 1000):
    try:
        # 初始化 Binance 交易所
        exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
            'proxies': {
                'http': 'http://127.0.0.1:7890',  # 设置 HTTP 代理
                'https': 'http://127.0.0.1:7890'  # 设置 HTTPS 代理
            }
        })

        # 获取指定交易对的 K 线数据
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)

        # 转换为 Pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # 格式化时间戳为可读格式，并调整为东八区时间
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + timedelta(hours=8)

        print(f"成功获取 {pair} 的 {timeframe} 数据（时间已转换为东八区）：")
        print(df.head())
        '''计算涨跌幅'''
        df = calculate_price_change(df)

        # 保存为 CSV 文件
        output_file = fr"data\{pair.replace('/', '_')}_{timeframe}.csv"
        df.to_csv(output_file, index=False)
        print(f"数据已保存至 {output_file}")

        return df
    except ccxt.BaseError as e:
        print(f"获取数据时发生错误: {e}")
        return None


def calculate_price_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算收盘价的涨跌幅度 (百分比变化)

    参数:
        df (pd.DataFrame): 包含收盘价的 DataFrame，需包含 'close' 列

    返回:
        pd.DataFrame: 添加了 'price_change' 列的 DataFrame
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame 必须包含 'close' 列。")

    # 计算涨跌幅度，百分比表示
    df['price_change'] = df['close'].pct_change() * 100  # pct_change() 计算百分比变化
    return df


def calculate_final_price_change(file_path: str) -> float:
    """
    读取 CSV 数据文件并计算最终涨跌幅。

    参数:
        file_path (str): CSV 文件路径

    返回:
        float: 最终涨跌幅，百分比表示
    """
    try:
        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 检查是否包含 'close' 列
        if 'close' not in df.columns:
            raise ValueError(f"文件 {file_path} 中缺少 'close' 列。")

        # 获取第一个和最后一个收盘价
        first_close = df['close'].iloc[0]
        last_close = df['close'].iloc[-1]

        # 计算最终涨跌幅
        final_price_change = ((last_close - first_close) / first_close) * 100

        print(f"文件 {file_path} 的最终涨跌幅: {final_price_change:.2f}%")
        return final_price_change
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None


def process_all_csv_in_directory(directory: str):
    """
    遍历指定目录下的所有 CSV 文件，计算它们的最终涨跌幅。

    参数:
        directory (str): 包含 CSV 文件的目录路径
    """
    try:
        # 获取目录下的所有文件
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]

        if not files:
            print("指定目录中没有 CSV 文件。")
            return

        # 遍历所有 CSV 文件并计算最终涨跌幅
        for file_name in files:
            file_path = os.path.join(directory, file_name)
            calculate_final_price_change(file_path)
    except Exception as e:
        print(f"处理目录 {directory} 时出错: {e}")



# 示例：读取当前目录下所有 CSV 文件并计算涨跌幅
directory_path = "./data"  # 替换为你的 CSV 文件所在的目录
process_all_csv_in_directory(directory_path)






#
# '''选币列表'''
# bi_list = {
#     'DOGE','PEOPLE','FLOKI','PEPE','DOGS','BOME'',SOL','ETH','BNB','BTC'
#
# }
# '''批量获取数据'''
# for bi in bi_list:
#     fetch_binance_ohlcv(f'{bi}/USDT', '15m', 1000)
#     print(f'{bi}/USDT')


# df = pd.read_csv(r'data\BTC_USDT_15m.csv')
# df = calculate_price_change(df)
# print(df['price_change'])



# 示例: 获取 BTC/USDT 的前 1000 条 15 分钟 K 线数据
# fetch_binance_ohlcv('BTC/USDT', '15m', 1000)
