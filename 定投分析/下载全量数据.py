import ccxt
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条

def fetch_binance_ohlcv(exchange,pair: str, timeframe: str = '15m', limit: int = 1000):
    try:

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
        output_file = fr"data_all\{pair.replace('/', '_')}_{timeframe}.csv"
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
        'http': 'http://127.0.0.1:7890',  # 设置 HTTP 代理
        'https': 'http://127.0.0.1:7890'  # 设置 HTTPS 代理
    }
})

# 获取所有市场
markets = exchange.fetch_markets()
df = pd.DataFrame(markets)
df_spot = df[(df['type'] == type1) & (df['quote'] == quote)]
data = df_spot['id'].tolist()

# 转换为格式化的列表
bi_list = [f"{item[:-4]}" for item in data]
print(bi_list)
print(f'类型为 {type1} 的币对名共有 {len(bi_list)} 个')

# 使用 tqdm 显示进度条
for bi in tqdm(bi_list, desc="Fetching data"):
    fetch_binance_ohlcv(exchange,f'{bi}/USDT', '15m', 1000)
    print(f'{bi}/USDT')
