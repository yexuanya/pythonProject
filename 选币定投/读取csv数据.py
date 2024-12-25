import pandas as pd
import os

def load_data_from_csv(folder_path: str, encoding: str = 'gbk') -> pd.DataFrame:
    """
    读取指定文件夹中的所有CSV文件，合并成一个包含币种和时间作为多级索引的DataFrame。

    参数:
    folder_path: str
        存放CSV文件的文件夹路径
    encoding: str, 默认为 'gbk'
        CSV文件的编码格式

    返回:
    pd.DataFrame
        合并后的DataFrame，索引为币种和时间
    """
    # 读取文件夹中所有的CSV文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 创建一个空的列表用于存储所有读取的DataFrame
    dfs = []

    # 遍历每个CSV文件
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, encoding=encoding, skiprows=1)

        # 获取币种名称（从文件名中提取，例如 'BTC-USDT' => 'BTC'）
        coin_name = file.split('-')[0]

        # 在数据框中添加一个新列 '币种'，其值为币种名称
        df['币种'] = coin_name

        # 将时间列转换为datetime类型（假设时间列名为 'candle_begin_time'）
        df['时间'] = pd.to_datetime(df['candle_begin_time'])

        # 将时间和币种列设置为多级索引
        df.set_index(['币种', '时间'], inplace=True)

        # 将数据框加入到列表中
        dfs.append(df)

    # 合并所有的DataFrame，合并后的DataFrame会按时间和币种进行索引
    result = pd.concat(dfs)

    return result

# 使用函数读取数据
folder_path = r'D:\code\1\pythonProject\数据\xbx\swap'
result = load_data_from_csv(folder_path)

# 输出结果
print(result)
