import pandas as pd
import numpy as np
from pyecharts.charts import Line
from pyecharts import options as opts


def buy_low_sell_high_strategy(
        df, interval_days=30, base_investment_amount=100, adjustment_threshold=0.01, sell_threshold=0.01
):
    """
    策略：价格下跌买入，价格上涨卖出。
    :param df: 包含历史数据的 DataFrame，至少包含 'close' 和 'timestamp' 列
    :param interval_days: 操作的时间间隔（天数）
    :param base_investment_amount: 基础金额（用于买入）
    :param adjustment_threshold: 调整阈值（基于均价和市值差异的百分比，用于买入）
    :param sell_threshold: 卖出阈值（基于均价和市值差异的百分比，用于卖出）
    :return: 带有策略计算结果的 DataFrame
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['cumulative_investment'] = 0.0  # 累计投资金额
    df['cumulative_shares'] = 0.0  # 累计购买的份额
    df['portfolio_value'] = 0.0  # 投资组合总价值
    df['7_day_avg'] = df['close'].rolling(window=24*7).mean()  # 计算7天均价，1天=24

    # 初始累计值
    cumulative_investment = 0.0
    cumulative_shares = 0.0
    last_trade_date = df['timestamp'].iloc[0] - pd.Timedelta(days=interval_days)

    for i, row in df.iterrows():
        current_date = row['timestamp']
        current_price = row['close']
        avg_7_day_price = row['7_day_avg']

        # 跳过没有足够数据计算均价的行
        if pd.isna(avg_7_day_price):
            continue

        # 每隔 interval_days 进行操作
        if (current_date - last_trade_date).days >= interval_days:
            # 计算与均价的偏差
            price_difference_ratio = (avg_7_day_price - current_price) / current_price

            # 买入条件：价格低于均价（下跌）
            if price_difference_ratio < 0 and abs(price_difference_ratio) >= adjustment_threshold:
                # 调整后的买入金额
                adjusted_investment = base_investment_amount * (1 + price_difference_ratio * 10)
                shares_bought = adjusted_investment / current_price
                cumulative_investment += adjusted_investment
                cumulative_shares += shares_bought
                last_trade_date = current_date  # 更新上次交易日期

            # 卖出条件：价格高于均价（上涨）
            elif price_difference_ratio > 0 and abs(price_difference_ratio) >= sell_threshold:
                # 计算卖出的份额（假设卖出10%的持仓）
                shares_sold = cumulative_shares * 0.1
                sell_amount = shares_sold * current_price
                cumulative_shares -= shares_sold
                # 不直接减少投资金额，避免为负，而是更新市值
                cumulative_investment = cumulative_shares * current_price
                last_trade_date = current_date  # 更新上次交易日期

        # 更新 DataFrame 中的累计值
        df.at[i, 'cumulative_investment'] = cumulative_investment
        df.at[i, 'cumulative_shares'] = cumulative_shares

        # 更新投资组合总价值
        df.at[i, 'portfolio_value'] = cumulative_shares * current_price

    # 计算净利润：市值 - 投入金额
    df['net_profit'] = df['portfolio_value'] - df['cumulative_investment']

    return df


def backtest_dca(df, interval_days=30, investment_amount=100):
    """
    回测定投策略
    :param df: 包含历史数据的 DataFrame
    :param interval_days: 定投的时间间隔（天数）
    :param investment_amount: 每次投资的固定金额
    :return: 包含回测结果的 DataFrame
    """
    df = buy_low_sell_high_strategy(df, interval_days, investment_amount)
    df['returns'] = df['portfolio_value'].pct_change()
    return df


def line_charts_dca(df):
    """
    根据定投数据绘制折线图。
    :param df: 包含 'timestamp', 'close', 'portfolio_value' 等列的 DataFrame
    """
    c = Line(
        init_opts=opts.InitOpts(
            width="100%",
            height="600px",
            page_title="BTC/USDT 定投回测",
        )
    )

    # 设置 x 轴为时间戳
    c.add_xaxis(xaxis_data=df['timestamp'].dt.strftime('%Y-%m-%d').tolist())

    # 添加 y 轴数据
    c.add_yaxis(
        series_name="收盘价",
        y_axis=df['close'].tolist(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    c.add_yaxis(
        series_name="投资组合价值",
        y_axis=df['portfolio_value'].tolist(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    c.add_yaxis(
        series_name="累计投资金额",
        y_axis=df['cumulative_investment'].tolist(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    c.add_yaxis(
        series_name="净利润",
        y_axis=df['net_profit'].tolist(),
        label_opts=opts.LabelOpts(is_show=False),
    )

    # 数据项设置
    c.set_global_opts(
        title_opts=opts.TitleOpts(
            title='BTC/USDT 定投回测',
            pos_left="center",
            pos_top="5%",
            title_textstyle_opts=opts.TextStyleOpts(font_size=18),
        ),
        legend_opts=opts.LegendOpts(pos_top="10%"),
        tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='cross'),
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        datazoom_opts=[opts.DataZoomOpts()],
        xaxis_opts=opts.AxisOpts(name="时间", type_="category"),
        yaxis_opts=opts.AxisOpts(name="价格/价值", type_="value"),
    )

    return c


if __name__ == "__main__":
    # 加载数据
    filename = r'D:\code\1\pythonProject\数据\ccxt\swap\BTC-USDT_20-24_1h.csv'  # 替换成你的数据文件
    df = pd.read_csv(filename)

    df['timestamp_copy'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp_copy', inplace=True)  # 使用复制的时间戳列作为索引

    # 回测定投策略
    df = backtest_dca(df, interval_days=24, investment_amount=1000)

    # 可视化回测结果
    chart = line_charts_dca(df)
    chart.render(path='BTC_USDT_dca_backtest.html')
    print("单次回测完成，图表已保存为 BTC_USDT_dca_backtest.html")
