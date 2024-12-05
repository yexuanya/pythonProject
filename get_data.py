import ccxt
import pandas as pd
import time

def fetch_historical_data(exchange_id, symbol, timeframe, since, until):
    """
    获取指定时间段的历史数据
    :param exchange_id: 交易所 ID（如 'binance'）
    :param symbol: 交易对（如 'BTC/USDT'）
    :param timeframe: 时间间隔（如 '1m', '1h', '1d'）
    :param since: 起始时间戳（毫秒）
    :param until: 结束时间戳（毫秒）
    :return: DataFrame 格式的历史数据
    """
    # 创建交易所实例
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'rateLimit': 1200,
        'enableRateLimit': True,
        'proxies': {
            'http': 'http://127.0.0.1:15236',  # 设置 HTTP 代理
            'https': 'http://127.0.0.1:15236'  # 设置 HTTPS 代理
        }
    })

    all_data = []
    while since < until:
        try:
            # 获取历史数据
            data = exchange.fetch_ohlcv(symbol, timeframe, since)
            if not data:
                break

            all_data.extend(data)
            # 获取最后一个数据点的时间戳
            last_timestamp = data[-1][0]
            since = last_timestamp + exchange.parse_timeframe(timeframe) * 1000  # 下一个时间段
            time.sleep(exchange.rateLimit / 1000)  # 避免速率限制
        except ccxt.BaseError as e:
            print(f"Error: {e}")
            break

    # 转换为 DataFrame
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")  # 转换时间戳为日期格式
    return df


# 使用示例
if __name__ == "__main__":

    # 交易所、交易对和时间间隔
    exchange_id = "binance"
    symbol = "BTC/USDT"
    timeframe = "1h"  # 可用的时间间隔：'1m', '5m', '15m', '1h', '1d', 等

    # 时间范围（毫秒）
    start_time = int(pd.Timestamp("2020-01-01").timestamp() * 1000)  # 起始时间
    end_time = int(pd.Timestamp("2024-12-1").timestamp() * 1000)  # 结束时间

    # 获取数据
    df = fetch_historical_data(exchange_id, symbol, timeframe, start_time, end_time)

    # 保存为 CSV
    df.to_csv(r"BTCUSDT_20-24_1h.csv", index=False)
    print(df)
