import ccxt
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条
import os

def fetch_binance_ohlcv(exchange, pair: str, timeframe: str = '15m', limit: int = 1000, output_folder: str = "data"):
    """
    获取指定交易对的历史 K 线数据，并保存为 CSV 文件

    参数:
        exchange: CCXT 交易所对象
        pair (str): 交易对，例如 'BTC/USDT'
        timeframe (str): 时间间隔，例如 '1h', '15m'
        limit (int): 获取的数据条数
        output_folder (str): 保存文件的文件夹路径
    返回:
        pd.DataFrame: 包含历史 K 线数据的 DataFrame
    """
    try:
        # 获取指定交易对的 K 线数据
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)

        # 转换为 Pandas DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # 转换时间戳为东八区时间（北京时间）
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + timedelta(hours=8)

        # 计算涨跌幅
        df = calculate_price_change(df)

        # 检查并创建文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 保存文件为 CSV
        output_file = os.path.join(output_folder, f"{pair.replace('/', '_')}_{timeframe}.csv")
        df.to_csv(output_file, index=False)

        print(f"成功获取 {pair} 的 {timeframe} 数据，已保存至 {output_file}")
        return df

    except ccxt.BaseError as e:
        print(f"获取 {pair} 数据时发生错误: {e}")
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

# ======= 主程序部分 ========
'''
spot：现货
swap：永续合约
'''
type1 = 'swap'
quote = 'USDT'

# 初始化 Binance 交易所
exchange = ccxt.binance({
    'enableRateLimit': True,  # 启用速率限制
    'proxies': {
        'http': 'http://127.0.0.1:15236',  # 设置 HTTP 代理
        'https': 'http://127.0.0.1:15236'  # 设置 HTTPS 代理
    }
})

# 获取所有市场信息
markets = exchange.fetch_markets()
df = pd.DataFrame(markets)

# 筛选目标交易对
df_spot = df[(df['type'] == type1) & (df['quote'] == quote)]
data = df_spot['id'].tolist()

# 转换为格式化的列表
bi_list = [f"{item[:-4]}" for item in data]
print(bi_list)
print(f'类型为 {type1} 的币对名共有 {len(bi_list)} 个')

# 设置保存文件夹路径
output_folder = "data_all_1h"

# 使用 tqdm 显示进度条
for bi in tqdm(bi_list, desc="Fetching data"):
    fetch_binance_ohlcv(exchange, f'{bi}/USDT', '1h', 1000, output_folder)
    print(f'{bi}/USDT')
