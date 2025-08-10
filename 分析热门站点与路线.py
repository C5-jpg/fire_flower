# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_bike_data(file_path, top_n=10):
    """
    分析共享单车数据，找出热门站点和路线，并生成可视化图表。

    参数:
    file_path (str): CSV文件的路径。
    top_n (int): 要显示的热门项目数量。
    """
    # --- 1. 数据加载 ---
    print(f"正在从 '{file_path}' 加载数据...")
    try:
        # 使用 engine='python' 增强对复杂路径或特殊字符的兼容性
        df = pd.read_csv(file_path, engine='python')
        print("数据加载成功！")
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'。请检查文件路径是否正确。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # --- 2. 数据处理与分析 ---
    
    # a. 分析最热门的出发站点
    print(f"正在分析排名前 {top_n} 的热门出发站点...")
    top_start_stations = df['start_station_name'].value_counts().nlargest(top_n)

    # b. 分析最热门的到达站点
    print(f"正在分析排名前 {top_n} 的热门到达站点...")
    top_end_stations = df['end_station_name'].value_counts().nlargest(top_n)

    # c. 分析最热门的路线
    print(f"正在分析排名前 {top_n} 的热门路线...")
    # 创建一个 'route' 列来表示从起点到终点的完整路线
    df['route'] = df['start_station_name'] + ' -> ' + df['end_station_name']
    top_routes = df['route'].value_counts().nlargest(top_n)

    # --- 3. 可视化 ---
    print("正在生成可视化图表...")
    
    # 设置matplotlib以正确显示中文（或其他非ASCII字符）
    # 'SimHei' 是一个常用的支持中文的字体，您的系统需要安装此字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建一个大画布，包含3个子图
    fig, axes = plt.subplots(3, 1, figsize=(12, 22))
    fig.suptitle('共享单车热门站点与路线分析', fontsize=20, y=0.95)

    # 图1: 热门出发站点
    sns.barplot(x=top_start_stations.values, y=top_start_stations.index, ax=axes[0], palette='viridis')
    axes[0].set_title(f'排名前 {top_n} 的热门出发站点', fontsize=16)
    axes[0].set_xlabel('骑行次数', fontsize=12)
    axes[0].set_ylabel('站点名称', fontsize=12)

    # 图2: 热门到达站点
    sns.barplot(x=top_end_stations.values, y=top_end_stations.index, ax=axes[1], palette='plasma')
    axes[1].set_title(f'排名前 {top_n} 的热门到达站点', fontsize=16)
    axes[1].set_xlabel('骑行次数', fontsize=12)
    axes[1].set_ylabel('站点名称', fontsize=12)

    # 图3: 热门路线
    sns.barplot(x=top_routes.values, y=top_routes.index, ax=axes[2], palette='magma')
    axes[2].set_title(f'排名前 {top_n} 的热门路线', fontsize=16)
    axes[2].set_xlabel('骑行次数', fontsize=12)
    axes[2].set_ylabel('路线', fontsize=12)

    # 调整布局以防标签重叠
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- 4. 保存图表 ---
    output_filename = 'popular_stations_and_routes.png'
    plt.savefig(output_filename)
    print(f"分析完成！图表已保存为 '{output_filename}'。")
    
    # 显示图表（如果您在本地环境中运行，会弹出一个窗口）
    plt.show()


if __name__ == '__main__':
    # 请将这里的路径替换为您自己文件的实际路径
    # Windows路径建议使用r''字符串，以避免反斜杠'\'被误解
    file_path = r"D:\wechat\xwechat_files\wxid_5ex8kjms8fa022_408c\msg\file\2025-08\2025年度“火花杯”数学建模精英联赛-B题-附件\2025年度“火花杯”数学建模精英联赛-B题-附件\demand_features.csv"
    
    # 检查文件是否存在
    if os.path.exists(file_path):
        analyze_bike_data(file_path)
    else:
        print(f"错误：文件路径 '{file_path}' 不存在。")
        print("请打开脚本文件，并将变量 'file_path' 的值修改为您的CSV文件的正确路径。")

