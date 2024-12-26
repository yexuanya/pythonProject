import pandas as pd
import os
import matplotlib.pyplot as plt
from pyecharts.charts import Line
from pyecharts import options as opts


def load_data_from_csv(folder_path: str, encoding: str = 'gbk') -> pd.DataFrame:
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []

    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, encoding=encoding, skiprows=1)

        coin_name = file.split('-')[0]
        df['币种'] = coin_name
        df['时间'] = pd.to_datetime(df['candle_begin_time'])
        df.set_index(['币种', '时间'], inplace=True)
        dfs.append(df)

    result = pd.concat(dfs)
    #输出文件，方便下次直接调用
    # result.to_csv('result.csv')
    return result


def get_data_by_time_and_coin(df: pd.DataFrame, start_date: str = None, end_date: str = None,
                              coins: list = None) -> pd.DataFrame:
    """
    获取指定时间范围和币种的数据

    参数：
    df: pd.DataFrame
        合并后的DataFrame
    start_date: str, 默认为 None
        开始日期 (格式：'YYYY-MM-DD')
    end_date: str, 默认为 None
        结束日期 (格式：'YYYY-MM-DD')
    coins: list, 默认为 None
        要筛选的币种列表，例如 ['BTC', 'ETH']

    返回：
    pd.DataFrame
        筛选后的DataFrame
    """
    # 筛选时间范围
    if start_date:
        df = df[df.index.get_level_values('时间') >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index.get_level_values('时间') <= pd.to_datetime(end_date)]

    # 筛选币种
    if coins:
        df = df[df.index.get_level_values('币种').isin(coins)]

    return df


# 获取每个币种的数据时间范围
def get_date_ranges_by_coin(df: pd.DataFrame) -> pd.DataFrame:
    """
    获取每个币种实际存在的数据时间范围

    参数：
    df: pd.DataFrame
        合并后的DataFrame

    返回：
    pd.DataFrame
        包含每个币种的起始和结束日期
    """
    # 重置索引方便分组
    filtered_df = df.reset_index()

    # 筛选非空的时间数据
    date_ranges = (
        filtered_df[~filtered_df['时间'].isna()]  # 排除时间为空的数据
        .groupby('币种')['时间']
        .agg(['min', 'max'])  # 获取最小值和最大值
        .rename(columns={'min': '开始日期', 'max': '结束日期'})  # 重命名列
    )

    return date_ranges


def calculate_investment_curve(df: pd.DataFrame, coins: list, start_date: str, end_date: str, amount: float = 100) -> pd.DataFrame:
    """
    模拟每天 4 点定投指定币种的收益曲线

    参数：
    df: pd.DataFrame
        合并后的数据框
    coins: list
        要定投的币种列表，例如 ['BTC', 'ETH']
    start_date: str
        定投开始日期 (格式：'YYYY-MM-DD')
    end_date: str
        定投结束日期 (格式：'YYYY-MM-DD')
    amount: float, 默认为 100
        每天每个币种定投的金额

    返回：
    pd.DataFrame
        包含日期和收益曲线的 DataFrame
    """
    # 筛选指定时间范围和币种的数据
    filtered_df = get_data_by_time_and_coin(df, start_date=start_date, end_date=end_date, coins=coins)

    # 筛选每天 4 点的数据，按日期聚合
    daily_data = filtered_df[filtered_df.index.get_level_values('时间').hour == 4]
    daily_data = daily_data.groupby(daily_data.index.get_level_values('时间').date).apply(
        lambda group: {coin: group.loc[group.index.get_level_values('币种') == coin, 'close'].iloc[0]
                       for coin in coins if coin in group.index.get_level_values('币种')}
    )

    # 初始化投资结果
    investment_results = []
    total_investment = {coin: 0 for coin in coins}
    total_holdings = {coin: 0 for coin in coins}

    for date, prices in daily_data.items():
        daily_investment = 0
        for coin, price in prices.items():
            if price > 0:  # 确保价格有效
                # 更新投资金额和持仓
                total_investment[coin] += amount
                total_holdings[coin] += amount / price
                daily_investment += amount

        # 计算当前总市值
        total_value = sum(total_holdings[coin] * prices.get(coin, 0) for coin in coins)
        total_cost = sum(total_investment.values())

        # 添加记录
        investment_results.append({
            '日期': date,
            '总市值': total_value,
            '总投入': total_cost,
            '收益': total_value - total_cost,
            '收益率': ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
        })

    # 转为 DataFrame
    investment_curve = pd.DataFrame(investment_results)

    return investment_curve



# plt绘制收益曲线
def plot_investment_curve(investment_curve: pd.DataFrame):
    """
    绘制收益曲线

    参数：
    investment_curve: pd.DataFrame
        包含收益曲线的 DataFrame
    """
    plt.figure(figsize=(12, 6))
    plt.plot(investment_curve['时间'], investment_curve['收益'], label='收益', color='blue')
    plt.plot(investment_curve['时间'], investment_curve['总市值'], label='总市值', color='green', linestyle='--')
    plt.plot(investment_curve['时间'], investment_curve['总投入'], label='总投入', color='red', linestyle=':')
    plt.plot(investment_curve['时间'], investment_curve['收益率'], label='收益率', color='purple', linestyle='-.')
    plt.title('每天 4 点定投 10 个币的收益曲线')
    plt.xlabel('时间')
    plt.ylabel('金额 (¥)')
    plt.legend()
    plt.grid()
    plt.show()


# pyecharts绘制收益曲线
def line_chart(investment_curve):
    """
    使用 pyecharts 绘制定投收益曲线图，包含收益率曲线。

    参数:
    investment_curve: pd.DataFrame
        包含时间、总市值、总投入、收益和收益率的 DataFrame。

    返回:
    pyecharts Line 图对象
    """
    # 确保日期列为 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(investment_curve['日期']):
        investment_curve['日期'] = pd.to_datetime(investment_curve['日期'])

    # 提取数据
    x = investment_curve['日期'].dt.strftime('%Y-%m-%d').tolist()  # 日期转为字符串
    total_investment = investment_curve['总投入'].tolist()
    total_value = investment_curve['总市值'].tolist()
    profit = investment_curve['收益'].tolist()
    profit_rate = investment_curve['收益率'].tolist()

    # 初始化 Line 图
    line = Line(init_opts=opts.InitOpts(width="100%", height="600px", page_title="定投收益曲线"))

    # 添加 x 轴
    line.add_xaxis(xaxis_data=x)

    # 添加金额相关的 Y 轴数据 (主 Y 轴)
    line.add_yaxis(series_name="总投入", y_axis=total_investment, is_smooth=True, yaxis_index=0)
    line.add_yaxis(series_name="总市值", y_axis=total_value, is_smooth=True, yaxis_index=0)
    line.add_yaxis(series_name="收益", y_axis=profit, is_smooth=True, yaxis_index=0)

    # 添加收益率数据 (次 Y 轴)
    line.add_yaxis(
        series_name="收益率 (%)",
        y_axis=profit_rate,
        is_smooth=True,
        yaxis_index=1,
        linestyle_opts=opts.LineStyleOpts(width=2, color="orange"),
        label_opts=opts.LabelOpts(is_show=False),
    )

    # 设置全局选项
    line.set_global_opts(
        title_opts=opts.TitleOpts(
            title="定投收益曲线",
            pos_left="center",
            pos_top="5%",
            title_textstyle_opts=opts.TextStyleOpts(font_size=18),
        ),
        legend_opts=opts.LegendOpts(is_show=True, pos_top="10%"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        datazoom_opts=[opts.DataZoomOpts(is_show=True, type_="slider", orient="horizontal")],
        yaxis_opts=opts.AxisOpts(
            name="金额 (¥)",
            position="left",
            axislabel_opts=opts.LabelOpts(formatter="{value}"),
        ),
        xaxis_opts=opts.AxisOpts(
            name="日期",
            axislabel_opts=opts.LabelOpts(rotate=45),
        ),
    )

    # 手动添加次 Y 轴
    line.set_series_opts(
        areastyle_opts=None,
        label_opts=opts.LabelOpts(is_show=False)
    )
    line.extend_axis(
        yaxis=opts.AxisOpts(
            name="收益率 (%)",
            position="right",
            axislabel_opts=opts.LabelOpts(formatter="{value}%"),
            splitline_opts=opts.SplitLineOpts(is_show=False),
        )
    )

    return line


def calculate_sharpe_ratio(investment_curve: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    """
    计算投资组合的夏普率 (Sharpe Ratio)

    参数：
    investment_curve: pd.DataFrame
        包含投资曲线的 DataFrame，必须包含 '总市值' 列。
    risk_free_rate: float, 默认为 0.02
        年化无风险收益率 (例如国债利率)，默认值为 2%。

    返回：
    float
        夏普率
    """
    # 确保按日期排序
    investment_curve = investment_curve.sort_values(by='日期')

    # 计算每日收益率
    investment_curve['每日收益率'] = investment_curve['总市值'].pct_change()

    # 转换无风险收益率为每日收益率
    daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 365) - 1

    # 计算超额收益率
    investment_curve['超额收益率'] = investment_curve['每日收益率'] - daily_risk_free_rate

    # 计算平均超额收益率和标准差
    mean_excess_return = investment_curve['超额收益率'].mean()
    std_dev_return = investment_curve['每日收益率'].std()

    # 计算夏普率
    sharpe_ratio = mean_excess_return / std_dev_return if std_dev_return != 0 else float('nan')

    return sharpe_ratio


# # 使用函数读取数据
# folder_path = r'D:\code\1\pythonProject\数据\xbx\spot'
# result = load_data_from_csv(folder_path)

#有完整数据可直接读取
result = pd.read_csv(
    'xbx-spot.csv',
    parse_dates=['时间'],  # 确保时间列正确解析为 datetime 格式
    index_col=['币种', '时间']  # 设置 MultiIndex
)

bi = ["BTC","ETH","SOL","XRP","TRX","DOGE","ADA","LINK","SUI","PEPE"]
# 获取 2020年11月1日到2020年11月10日的 BTC 和 ETH 数据
# filtered_data = get_data_by_time_and_coin(result, start_date='2024-01-01', end_date='2024-10-31', coins=bi)

# # 输出结果
# print(filtered_data)
# # 获取每个币种的日期范围
# date_ranges = get_date_ranges_by_coin(filtered_data)
# # 输出结果
# print(date_ranges)

# 计算收益曲线
investment_curve = calculate_investment_curve(
    df=result,
    coins=bi,
    start_date='2024-01-01',
    end_date='2024-11-30',
    amount=10
)


# 绘制图表
chart = line_chart(investment_curve)
chart.render(path="investment_curve.html")

sharpe_ratio = calculate_sharpe_ratio(investment_curve)
print(f"夏普率为: {sharpe_ratio:.4f}")