import pandas as pd
from pyecharts.charts import Line
from pyecharts import options as opts


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
    c.add_yaxis(
        series_name="收盘价",
        y_axis=df['close'].tolist(),
        # is_smooth=True,  # 设置为平滑曲线
        label_opts=opts.LabelOpts(is_show=False),  # 隐藏点的标签
    )
    c.add_yaxis(
        series_name="最高价",
        y_axis=df['high'].tolist(),
        # is_smooth=True,  # 设置为平滑曲线
        label_opts=opts.LabelOpts(is_show=False),  # 隐藏点的标签
    )
    c.add_yaxis(
        series_name="最低价",
        y_axis=df['low'].tolist(),
        # is_smooth=True,  # 设置为平滑曲线
        label_opts=opts.LabelOpts(is_show=False),  # 隐藏点的标签
    )
    c.add_yaxis(
        series_name="成交量",
        y_axis=df['volume'].tolist(),
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


# 示例数据
# data = {
#     "timestamp": ["2023-12-01", "2023-12-02", "2023-12-03"],  # 替换为实际数据
#     "close": [45000, 46000, 47000],  # 替换为实际收盘价
# }
# df = pd.DataFrame(data)
if __name__ == '__main__':
    df = pd.read_csv('btcusdt_data.csv')

    # 绘制图表
    chart = line_charts(df)
    chart.render(path='BTC_USDT_close_price_trend.html')
