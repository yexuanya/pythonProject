from pyecharts.charts import Line
from pyecharts import options as opts


def line_charts():
    c = Line(
        init_opts=opts.InitOpts(
            width="100%",  # 设置图表宽度为100%页面宽度
            height="600px",  # 设置图表高度
            page_title="原力增长统计图",  # 设置页面标题
        )
    )
    # 设置x轴
    c.add_xaxis(xaxis_data=x)
    # 设置y轴
    c.add_yaxis(series_name='博主A', y_axis=y1)
    c.add_yaxis(series_name='博主B', y_axis=y2)
    c.add_yaxis(series_name='博主C', y_axis=y3)

    # 数据项设置
    data_zoom = {
        "show": True,
        "title": {"缩放": "数据缩放", "还原": "缩放数据还原"}
    }
    c.set_global_opts(
        title_opts=opts.TitleOpts(
            title='博主年末后三个月原力增长数量',
            pos_left="center",
            pos_top="5%",  # 设置标题距离顶部5%
            title_textstyle_opts=opts.TextStyleOpts(font_size=18),  # 调整标题字体大小
        ),
        legend_opts=opts.LegendOpts(
            is_show=True,
            pos_top="10%"  # 设置图例距离顶部的距离，避免与标题重叠
        ),
        tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='cross'),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            orient='horizontal',
            feature=opts.ToolBoxFeatureOpts(data_zoom=data_zoom)
        )
    )

    return c


# X轴数据
x = ['10月份', '11月份', '12月份']

# Y轴数据
y1 = [1120, 520, 770]
y2 = [1000, 300, 800]
y3 = [1072, 500, 900]

# 绘制图表
c = line_charts()
c.render(path='Demo1_base_lineChart.html')
