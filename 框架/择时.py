import pandas as pd
import numpy as np
from pyecharts.charts import Line
from pyecharts import options as opts


def rsi_strategy(df, period=14, buy_threshold=30, sell_threshold=70, base_investment_amount=1000):
    """
    基于RSI的择时策略：超卖时买入，超买时卖出。
    :param df: 包含历史数据的 DataFrame，至少包含 'close' 和 'timestamp' 列
    :param period: RSI的计算周期
    :param buy_threshold: 买入阈值，RSI低于该值时买入
    :param sell_threshold: 卖出阈值，RSI高于该值时卖出
    :param base_investment_amount: 基础金额（初始资金）
    :return: 带有策略计算结果的 DataFrame
    """
    # 计算RSI
    df['price_diff'] = df['close'].diff()  # 收盘价的差值(和上一条记录的价差)
    df['gain'] = np.where(df['price_diff'] > 0, df['price_diff'], 0)
    df['loss'] = np.where(df['price_diff'] < 0, -df['price_diff'], 0)

    # 计算滚动平均的增益和损失
    df['avg_gain'] = df['gain'].rolling(window=period).mean()
    df['avg_loss'] = df['loss'].rolling(window=period).mean()

    # 计算RSI
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['rsi'] = 100 - (100 / (1 + df['rs']))

    # 初始化策略相关列
    df['cumulative_shares'] = 0.0  # 累计购买的份额
    df['portfolio_value'] = 0.0  # 投资组合总价值
    df['trade_signal'] = 0  # 交易信号：1 = 买入，-1 = 卖出，0 = 无操作

    # 初始累计值
    cumulative_shares = 0.0
    cash_balance = base_investment_amount  # 用初始资金作为现金余额

    # 执行策略
    for i, row in df.iterrows():
        current_price = row['close']
        current_rsi = row['rsi']

        # 买入条件：RSI低于买入阈值，且有足够现金
        if current_rsi < buy_threshold :
            # 计算买入的份额
            shares_bought = cash_balance / current_price  # 取整，不能购买部分份额
            cash_balance -= shares_bought * current_price  # 扣除现金
            cumulative_shares += shares_bought
            df.at[i, 'trade_signal'] = 1  # 记录买入信号

        # 卖出条件：RSI高于卖出阈值，且有持仓
        elif current_rsi > sell_threshold and cumulative_shares > 0:
            # 卖出所有持仓
            sell_amount = cumulative_shares * current_price
            cash_balance += sell_amount  # 卖出后得到现金
            cumulative_shares = 0  # 清空持仓
            df.at[i, 'trade_signal'] = -1  # 记录卖出信号

        # 更新投资组合总价值
        df.at[i, 'portfolio_value'] = cumulative_shares * current_price + cash_balance

    return df



def backtest_rsi(df, period=14, buy_threshold=30, sell_threshold=70, investment_amount=100):
    """
    回测RSI择时策略
    :param df: 包含历史数据的 DataFrame
    :param period: RSI计算周期
    :param buy_threshold: 买入阈值
    :param sell_threshold: 卖出阈值
    :param investment_amount: 每次买入或卖出的金额
    :return: 带有策略计算结果的 DataFrame
    :cumulative_returns：回测RSI择时策略，计算累计收益。
    """
    df = rsi_strategy(df, period, buy_threshold, sell_threshold, investment_amount)
    df['returns'] = df['portfolio_value'].pct_change()  # 计算每日收益率

    return df


def line_charts_rsi(df):
    """
    根据RSI择时策略绘制折线图，包括累计收益曲线。
    :param df: 包含 'timestamp', 'close', 'portfolio_value', 'rsi', 'cumulative_returns' 等列的 DataFrame
    """
    c = Line(
        init_opts=opts.InitOpts(
            width="100%",
            height="600px",
            page_title="RSI择时策略回测",
        )
    )

    # 设置 x 轴为时间戳
    c.add_xaxis(xaxis_data=df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist())

    # 添加 y 轴数据
    c.add_yaxis(
        series_name="收盘价",
        y_axis=df['close'].tolist(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    c.add_yaxis(
        series_name="资金曲线",
        y_axis=df['portfolio_value'].tolist(),  # 使用原始累计收益（未乘以100）
        label_opts=opts.LabelOpts(is_show=False),
    )

    # 数据项设置
    c.set_global_opts(
        title_opts=opts.TitleOpts(
            title='RSI择时策略回测',
            pos_left="center",
            pos_top="5%",
            title_textstyle_opts=opts.TextStyleOpts(font_size=18),
        ),
        legend_opts=opts.LegendOpts(pos_top="10%"),
        tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='cross'),
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        datazoom_opts=[opts.DataZoomOpts()],
        xaxis_opts=opts.AxisOpts(name="时间", type_="category"),
        yaxis_opts=opts.AxisOpts(name="价格/价值", type_="value")
    )

    return c



if __name__ == "__main__":
    # 加载数据
    filename = r'D:\code\1\pythonProject\数据\ccxt\swap\BTC-USDT_20-24_1h.csv'  # 替换成你的数据文件
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 回测RSI择时策略
    df = backtest_rsi(df, period=14, buy_threshold=30, sell_threshold=70, investment_amount=10000)

    # 输出详细数据
    df.to_csv('RSI_strategy_backtest.csv', index=False)

    # 可视化回测结果
    chart = line_charts_rsi(df)
    chart.render(path='RSI_strategy_backtest.html')
    print("RSI择时策略回测完成，图表已保存为 RSI_strategy_backtest.html")
