import ccxt
import time
import json
import gym
import numpy as np
import pandas as pd
from collections import deque
import random


# 初始化 Binance 交易所
exchange = ccxt.binance({
    'enableRateLimit': True,  # 启用速率限制
    'proxies': {
        'http': 'http://127.0.0.1:15236',  # 设置 HTTP 代理
        'https': 'http://127.0.0.1:15236'  # 设置 HTTPS 代理
    }
})


# 持续获取 BTC/USDT 实时数据，每秒更新一次
def fetch_realtime_data(symbol, interval=1, filename="data.jsonl"):
    """
    持续获取实时数据
    :param symbol: 交易对，例如 'BTC/USDT'
    :param interval: 每次请求的时间间隔，单位为秒
    """
    data_list = []  # 用于存储获取的数据

    try:
        while True:
            ticker = exchange.fetch_ticker(symbol)
            ticker_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'high' : ticker['high'],
                'low' : ticker['low'],
                'bid' : ticker['bid'],
                'bidVolume' : ticker['bidVolume'],
                'ask' : ticker['ask'],
                'askVolume' : ticker['askVolume'],
                'last' : ticker['last'],
                'percentage' : ticker['percentage'],
                'open' : ticker['open'],
                'close' : ticker['close'],
                'quoteVolume' : ticker['quoteVolume']
            }
            # 获取实时数据
            print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            # print(f"最高价格: {ticker['high']} USDT")
            # print(f"最低价格: {ticker['low']} USDT")
            # print(f"最高买单价: {ticker['bid']} USDT")
            # print(f"最高买单价数量: {ticker['bidVolume']}")
            # print(f"最低买单价: {ticker['ask']} USDT")
            # print(f"最低买单价数量: {ticker['askVolume']}")
            print(f"最近成交价: {ticker['last']} USDT")
            # print(f"价格变化: {ticker['percentage']}")
            # print(f"开盘价: {ticker['open']} USDT")
            # print(f"收盘价: {ticker['close']} USDT")
            # print(f"当前时间重交易量: {ticker['quoteVolume']}")
            # print("-" * 30)
            data_list.append(ticker_data)

            with open(filename, 'a', encoding='utf-8') as f:
                json.dump(ticker_data, f)
                f.write("\n")  # 每条数据单独一行

            # 等待指定时间间隔
            time.sleep(interval)
    except KeyboardInterrupt:
        print("手动停止循环")
    except Exception as e:
        print(f"发生错误: {e}")


# def Strategy (ticker):


# 每秒更新 BTC/USDT 数据
fetch_realtime_data('BTC/USDT', interval=1)


