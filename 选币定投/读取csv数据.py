import pandas as pd
import os

# 读取文件夹中所有的CSV文件
folder_path = r'D:\code\1\pythonProject\数据\xbx\swap'
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 创建一个空的列表用于存储所有读取的DataFrame
dfs = []

# 遍历每个CSV文件
for file in files:
    # 读取CSV文件
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path,encoding='gbk', skiprows=1)

    # 获取币种名称（从文件名中提取，例如 'BTC-USDT' => 'BTC'）
    coin_name = file.split('-')[0]

    # 在数据框中添加一个新列 '币种'，其值为币种名称
    df['币种'] = coin_name

    # 将时间列转换为datetime类型（假设时间列名为 '时间'，根据实际情况调整）
    df['时间'] = pd.to_datetime(df['candle_begin_time'])

    # 将时间和币种列设置为多级索引
    df.set_index(['币种', '时间'], inplace=True)

    # 将数据框加入到列表中
    dfs.append(df)

# 合并所有的DataFrame，合并后的DataFrame会按时间和币种进行索引
result = pd.concat(dfs)

# 现在你可以使用币种和时间进行索引访问
print(result)
