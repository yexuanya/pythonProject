import pandas as pd
import os
from datetime import datetime

def calculate_price_changes(file_path: str):
    """
    读取 CSV 数据文件并计算最高涨幅、最低涨幅、最终涨跌幅和平均涨幅。

    参数:
        file_path (str): CSV 文件路径

    返回:
        dict: 包含币种名称、最高涨幅、最低涨幅、最终涨幅和平均涨幅的字典
    """
    try:
        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 检查是否包含 'close' 列
        if 'close' not in df.columns:
            raise ValueError(f"文件 {file_path} 中缺少 'close' 列。")

        # 获取币种名称（从文件名提取）
        coin_name = os.path.basename(file_path).replace('.csv', '')

        # 计算涨跌幅
        df['price_change'] = df['close'].pct_change() * 100

        # 计算最高涨幅和最低涨幅
        max_increase = df['price_change'].max()  # 最高涨幅
        max_decrease = df['price_change'].min()  # 最低涨幅

        # 计算平均涨幅
        average_change = df['price_change'].mean()  # 平均涨幅

        # 获取第一个和最后一个收盘价，计算最终涨幅
        first_close = df['close'].iloc[0]
        last_close = df['close'].iloc[-1]
        final_price_change = ((last_close - first_close) / first_close) * 100

        # 返回结果
        return {
            'Coin': coin_name,
            'Max Increase (%)': max_increase,
            'Max Decrease (%)': max_decrease,
            'Final Change (%)': final_price_change,
            'Average Change (%)': average_change
        }
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None


def process_all_csv_in_directory(directory: str, output_dir: str):
    """
    遍历指定目录下的所有 CSV 文件，计算它们的涨幅信息并保存到表格。

    参数:
        directory (str): 包含 CSV 文件的目录路径
        output_dir (str): 输出结果保存的目录路径
    """
    try:
        # 获取目录下的所有 CSV 文件
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]

        if not files:
            print("指定目录中没有 CSV 文件。")
            return

        results = []

        # 遍历所有 CSV 文件并计算涨幅信息
        for file_name in files:
            file_path = os.path.join(directory, file_name)
            result = calculate_price_changes(file_path)
            if result:
                results.append(result)

        # 转换为 DataFrame
        result_df = pd.DataFrame(results)

        # 获取今天的日期并格式化为 mm_dd
        today = datetime.today().strftime('%m_%d')

        # 生成包含日期的输出文件名
        output_file = os.path.join(output_dir, f"summary_{today}.csv")

        # 保存结果为 CSV 文件
        result_df.to_csv(output_file, index=False)
        print(f"所有结果已保存至 {output_file}")

    except Exception as e:
        print(f"处理目录 {directory} 时出错: {e}")


# 示例：读取当前目录下所有 CSV 文件并计算涨幅信息
directory_path = "./data_all"  # 替换为你的 CSV 文件所在的目录
output_dir = "./"  # 替换为你的输出文件所在目录
process_all_csv_in_directory(directory_path, output_dir)
