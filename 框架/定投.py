import pandas as pd
import numpy as np
from pyecharts.charts import Line
from pyecharts import options as opts


def adjusted_dollar_cost_averaging_strategy(df, interval_days=30, base_investment_amount=100, adjustment_threshold=0.5):
    """
    定投策略：每隔一定时间间隔，定期投资基础金额，并根据市值与均价差异调整投资金额。
    定投添加波动系数，超过均价降低投入，低于均价加大投入
    :param df: 包含历史数据的 DataFrame，至少包含 'close' 和 'timestamp' 列
    :param interval_days: 定投的时间间隔（天数）
    :param base_investment_amount: 基础定投金额
    :param adjustment_threshold: 调整阈值（基于均价和市值差异的百分比）
    :return: 带有策略计算结果的 DataFrame
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['cumulative_investment'] = 0.0  # 累计投资金额
    df['cumulative_shares'] = 0.0      # 累计购买的份额
    df['portfolio_value'] = 0.0        # 投资组合总价值
    df['7_day_avg'] = df['close'].rolling(window=720).mean()  # 计算7天均价

    # 初始累计值
    cumulative_investment = 0.0
    cumulative_shares = 0.0
    last_investment_date = df['timestamp'].iloc[0] - pd.Timedelta(days=interval_days)   #记录上次投资日期

    for i, row in df.iterrows():
        current_date = row['timestamp']
        current_price = row['close']
        avg_7_day_price = row['7_day_avg']

        # 跳过没有足够数据计算均价的行
        if pd.isna(avg_7_day_price):
            continue

        # 每隔 interval_days 投资一次
        if (current_date - last_investment_date).days >= interval_days:
            # 计算投资调整比例
            price_difference_ratio = (avg_7_day_price - current_price) / current_price

            # 限制调整比例在 [-adjustment_threshold, adjustment_threshold] 范围内
            adjustment_ratio = max(min(price_difference_ratio, adjustment_threshold), -adjustment_threshold)

            # 调整后的投资金额
            adjusted_investment = base_investment_amount * (1 + adjustment_ratio*10)

            # 购买份额
            shares_bought = adjusted_investment / current_price
            cumulative_investment += adjusted_investment
            cumulative_shares += shares_bought
            last_investment_date = current_date

        # 更新 DataFrame 中的累计值
        df.at[i, 'cumulative_investment'] = cumulative_investment
        df.at[i, 'cumulative_shares'] = cumulative_shares

        # 更新投资组合总价值
        df.at[i, 'portfolio_value'] = cumulative_shares * current_price

    return df



def backtest_dca(df, interval_days=30, investment_amount=100):
    """
    回测定投策略
    :param df: 包含历史数据的 DataFrame
    :param interval_days: 定投的时间间隔（天数）
    :param investment_amount: 每次投资的固定金额
    :return: 包含回测结果的 DataFrame
    :adjusted_dollar_cost_averaging_strategy:根据波动调整投入比例
    :buy_low_sell_high_strategy:：根据波动买入卖出
    """
    df = adjusted_dollar_cost_averaging_strategy(df, interval_days, investment_amount)

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
    '''xxct数据'''
    filename = r'D:\code\1\pythonProject\数据\ccxt\swap\BTC-USDT_20-24_1h.csv'  # 替换成你的数据文件
    df = pd.read_csv(filename)

    '''xbx数据'''
    # filename = r'D:\code\1\pythonProject\数据\xbx\swap\BTC-USDT.csv'  # 替换成你的数据文件
    # df = pd.read_csv(filename, skiprows=1,encoding='GB2312')
    # df['timestamp'] = df['candle_begin_time']


    # df['timestamp_copy'] = df['timestamp']  # 将时间戳列复制一份
    df['timestamp_copy'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp_copy', inplace=True)  # 使用复制的时间戳列作为索引


    '''单次运行'''
    # 回测定投策略
    df = backtest_dca(df, interval_days=24, investment_amount=1000)

    # 可视化回测结果
    chart = line_charts_dca(df)
    chart.render(path='BTC_USDT_dca_backtest.html')
    print("单次回测完成，图表已保存为 BTC_USDT_dca_backtest.html")
