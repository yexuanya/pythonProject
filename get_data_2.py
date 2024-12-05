import ccxt
import pandas as pd

from get_data import fetch_historical_data

# 初始化 Binance 交易所
exchange = ccxt.binance({
    'enableRateLimit': True,  # 启用速率限制
    'proxies': {
        'http': 'http://127.0.0.1:15236',  # 设置 HTTP 代理
        'https': 'http://127.0.0.1:15236'  # 设置 HTTPS 代理
    }
})

# # 创建一个交易所实例，例如 binance
# exchange = ccxt.binance()

def print_now(row):
    print(f"比对名：{row['symbol']}，所属类型：{row['type']}")


def doload(row):
    # 交易所、交易对和时间间隔
    exchange_id = "binance"
    symbol = row['symbol']
    timeframe = "1h"  # 可用的时间间隔：'1m', '5m', '15m', '1h', '1d', 等

    # 时间范围（毫秒）
    start_time = int(pd.Timestamp("2020-01-01").timestamp() * 1000)  # 起始时间
    end_time = int(pd.Timestamp("2024-12-1").timestamp() * 1000)  # 结束时间

    # 获取数据
    df = fetch_historical_data(exchange_id, symbol, timeframe, start_time, end_time)

    # 保存为 CSV
    df.to_csv(fr"D:\code\1\pythonProject\数据\ccxt\swap\{row['base']}-{row['quote']}_20-24_1h.csv", index=False)
    print(symbol)

'''
spot：现货
swap：永续合约
'''
type1 = 'swap'
quote = 'USDT'


# 获取所有市场
markets = exchange.fetch_markets()
df = pd.DataFrame(markets)
df_spot = df[(df['type'] == type1) & (df['quote'] == quote)]

# df_spot.apply(print_now, axis=1)
df_spot.apply(doload, axis=1)

print(f'类型为 {type1} 的币对名共有 {len(df_spot)} 个')
