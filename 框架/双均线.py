import pandas as pd
import numpy as np
from pyecharts.charts import Line
from pyecharts import options as opts
from tqdm import tqdm  # 引入 tqdm

def moving_average_crossover_strategy(df, short_window=1, long_window=21):
    """
    双均线策略：计算短期和长期移动平均线的交叉信号
    """
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()
    df['buy_signal'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
    df['sell_signal'] = np.where(df['short_ma'] < df['long_ma'], -1, 0)
    df['signal'] = df['buy_signal'] + df['sell_signal']
    df['position'] = df['signal']
    return df


def backtest_strategy(df, initial_capital=10000, transaction_fee=0.001):
    """
    回测双均线策略，基于资金百分比买入卖出，并考虑手续费。
    """
    capital = initial_capital
    positions = 0
    capital_history = []

    for i in range(1, len(df)):
        if df['position'].iloc[i] == 1 and capital > 0:  # 买入信号
            capital_to_use = capital
            positions_to_buy = capital_to_use * (1 - transaction_fee) / df['close'].iloc[i]
            positions += positions_to_buy
            capital -= capital_to_use
        elif df['position'].iloc[i] == -1 and positions > 0:  # 卖出信号
            capital_to_use = positions
            positions_to_sell = capital_to_use
            positions -= positions_to_sell
            capital += positions_to_sell * df['close'].iloc[i] * (1 - transaction_fee)

        capital_history.append(capital + positions * df['close'].iloc[i])

    df['capital'] = [initial_capital] + capital_history
    df['returns'] = df['capital'].pct_change()
    return df


def optimize_parameters(df, max_short_window=50, max_long_window=50):
    """
    遍历多个窗口参数，找到最优参数并保存结果。
    """
    results = []

    # 使用 tqdm 包装原来的循环，添加进度条
    for n1 in tqdm(range(1, max_short_window), desc="优化短期窗口", unit="短期窗口"):
        for n2 in range(n1, max_long_window):
            df1 = moving_average_crossover_strategy(df, short_window=n1, long_window=n2)
            df1 = backtest_strategy(df1)
            final_capital = df1['capital'].iloc[-1]
            results.append([n1, n2, final_capital])

    results_df = pd.DataFrame(results, columns=['short_window', 'long_window', 'final_capital'])
    optimal_row = results_df.loc[results_df['final_capital'].idxmax()]

    # 保存结果
    results_df.to_csv('backtest_results.csv', index=False)
    print(f"最优参数：short_window={int(optimal_row['short_window'])}, "
          f"long_window={int(optimal_row['long_window'])}, "
          f"最终资本={optimal_row['final_capital']}")
    return optimal_row


def line_charts(df):
    """
    根据数据绘制折线图。
    """
    c = Line(
        init_opts=opts.InitOpts(width="100%", height="600px", page_title="BTC/USDT 收盘价趋势图")
    )
    c.add_xaxis(xaxis_data=df['timestamp'].tolist())
    c.add_yaxis(
        series_name="开盘价",
        y_axis=df['open'].tolist(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    c.add_yaxis(
        series_name="资金曲线",
        y_axis=df['capital'].tolist(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    c.set_global_opts(
        title_opts=opts.TitleOpts(title='BTC/USDT 收盘价趋势图', pos_left="center"),
        legend_opts=opts.LegendOpts(is_show=True),
        tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='cross'),
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        datazoom_opts=[opts.DataZoomOpts()],
        xaxis_opts=opts.AxisOpts(name="时间", type_="category"),
        yaxis_opts=opts.AxisOpts(name="价格", type_="value"),
    )
    return c


if __name__ == "__main__":
    # 加载数据
    filename = r'../btcusdt_data.csv'  # 替换成你的数据文件
    df = pd.read_csv(filename)

    # df['timestamp_copy'] = df['timestamp']  # 将时间戳列复制一份
    df['timestamp_copy'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp_copy', inplace=True)  # 使用复制的时间戳列作为索引


    '''单次运行'''
    # 使用默认参数单次回测并绘图
    short_window = 66  # 默认短期窗口
    long_window = 74   # 默认长期窗口
    df = moving_average_crossover_strategy(df, short_window=short_window, long_window=long_window)
    df = backtest_strategy(df)
    chart = line_charts(df)
    chart.render(path='BTC_USDT_close_price_trend.html')
    print("单次回测完成，图表已保存为 BTC_USDT_close_price_trend.html")


    '''遍历最优解'''
    # optimal_params = optimize_parameters(df, max_short_window=100, max_long_window=100)
    # print("最优参数已保存至 backtest_results2.csv")
