import pandas as pd
import numpy as np
from pyecharts.charts import Line
from pyecharts import options as opts

def moving_average_crossover_strategy(df, short_window=1, long_window=21):
    """
    双均线策略：计算短期和长期移动平均线的交叉信号
    :param df: 包含历史数据的 DataFrame，至少包含'close'列
    :param short_window: 短期均线窗口（天数）
    :param long_window: 长期均线窗口（天数）
    :return: 带有策略信号的 DataFrame
    """
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()
    df['buy_signal'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
    df['sell_signal'] = np.where(df['short_ma'] < df['long_ma'], -1, 0)
    df['signal'] = df['buy_signal'] + df['sell_signal']
    df['position'] = df['signal']
    return df


def backtest_strategy(df, initial_capital=10000, transaction_fee=0.002):
    """
    回测双均线策略，基于资金百分比买入卖出，并考虑手续费。
    :param df: 带有交易信号的 DataFrame
    :param initial_capital: 初始资本
    :param transaction_fee: 每笔交易的手续费比例，默认0.1%（0.002）
    :return: 包含资金曲线的 DataFrame
    """
    capital = initial_capital
    positions = 0
    capital_history = []

    for i in range(1, len(df)):
        # 买入信号
        if df['position'].iloc[i] == 1 and capital > 0:  # 如果当前是买入信号
            capital_to_use = capital
            positions_to_buy = capital_to_use * (1 - transaction_fee) / df['close'].iloc[i]
            positions += positions_to_buy
            capital -= capital_to_use   # 扣除手续费

        # 卖出信号
        if df['position'].iloc[i] == -1 and positions > 0:  # 如果当前是卖出信号
            capital_to_use = positions
            positions_to_sell = capital_to_use
            positions -= positions_to_sell
            capital += positions_to_sell * df['close'].iloc[i] - (positions_to_sell * df['close'].iloc[i] * transaction_fee)

        capital_history.append(capital + positions * df['close'].iloc[i])

    df['capital'] = [initial_capital] + capital_history
    df['returns'] = df['capital'].pct_change()
    return df


def save_backtest_results(df, max_short_window=50, max_long_window=50):
    """
    遍历多个窗口参数，保存每次回测的结果为 CSV 文件
    :param df: 输入的 DataFrame
    :param max_short_window: 短期窗口最大值
    :param max_long_window: 长期窗口最大值
    :return: None
    """
    results = []

    # 遍历所有窗口参数组合
    for n1 in range(1, max_short_window):
        for n2 in range(n1, max_long_window):
            df1 = moving_average_crossover_strategy(df, short_window=n1, long_window=n2)
            df1 = backtest_strategy(df1)

            final_capital = df1['capital'].iloc[-1]
            results.append([n1, n2, final_capital])

    # 保存为 CSV 文件
    results_df = pd.DataFrame(results, columns=['short_window', 'long_window', 'final_capital'])
    results_df.to_csv('backtest_results.csv', index=False)
    print('回测结果已保存到 backtest_results.csv 文件。')


def line_charts(df):
    """
    根据数据绘制折线图。
    :param df: 包含 'timestamp', 'close' 等列的 DataFrame
    """
    c = Line(
        init_opts=opts.InitOpts(
            width="100%",  # 设置图表宽度为100%页面宽度
            height="600px",  # 设置图表高度
            page_title="BTC/USDT 收盘价趋势图",  # 设置页面标题
        )
    )

    # 设置 x 轴为时间戳
    c.add_xaxis(xaxis_data=df['timestamp'].tolist())

    # 添加 y 轴为收盘价
    c.add_yaxis(
        series_name="开盘价",
        y_axis=df['open'].tolist(),
        # is_smooth=True,  # 设置为平滑曲线
        label_opts=opts.LabelOpts(is_show=False),  # 隐藏点的标签
    )
    # c.add_yaxis(
    #     series_name="收盘价",
    #     y_axis=df['close'].tolist(),
    #     # is_smooth=True,  # 设置为平滑曲线
    #     label_opts=opts.LabelOpts(is_show=False),  # 隐藏点的标签
    # )
    c.add_yaxis(
        series_name="资金曲线",
        y_axis=df['capital'].tolist(),
        # is_smooth=True,  # 设置为平滑曲线
        label_opts=opts.LabelOpts(is_show=False),  # 隐藏点的标签
    )

    # 数据项设置
    c.set_global_opts(
        title_opts=opts.TitleOpts(
            title='BTC/USDT 收盘价趋势图',
            pos_left="center",
            pos_top="5%",  # 设置标题距离顶部5%
            title_textstyle_opts=opts.TextStyleOpts(font_size=18),  # 调整标题字体大小
        ),
        legend_opts=opts.LegendOpts(
            is_show=True,
            pos_top="10%"  # 设置图例距离顶部的距离，避免与标题重叠
        ),
        tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='cross'),
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        datazoom_opts=[opts.DataZoomOpts()],  # 支持缩放
        xaxis_opts=opts.AxisOpts(name="时间", type_="category"),
        yaxis_opts=opts.AxisOpts(name="价格", type_="value"),
    )

    return c


if __name__ == "__main__":
    # 加载数据（这里假设数据为收盘价时间序列）
    filename = r'../btcusdt_data.csv'  # 替换成你的数据文件
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # 确保时间戳为 datetime 格式
    df.set_index('timestamp', inplace=True)

    # 执行回测结果保存
    save_backtest_results(df,max_short_window=500, max_long_window=500)

    # # 可视化回测结果
    # chart = line_charts(df)
    # chart.render(path='BTC_USDT_close_price_trend.html')
