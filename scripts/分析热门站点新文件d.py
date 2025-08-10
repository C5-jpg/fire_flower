# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm
import platform

# 确保中文显示正常的设置
def setup_chinese_fonts():
    # 常见的中文字体名称
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi', 
        'STHeiti', 'STSong', 'WenQuanYi Micro Hei', 'Heiti TC', 
        'Arial Unicode MS', 'Meiryo UI'
    ]
    
    # 查找系统中可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找可用字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            print(f"已设置中文字体: {font}")
            return True
    
    # 如果找不到中文字体，尝试使用matplotlib的默认字体配置
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    print("警告：未找到理想的中文字体，已尝试使用备选字体配置。")
    print("提示：如果中文仍显示异常，可能需要安装中文字体或检查字体文件权限。")
    return False

# 提前设置中文字体
setup_chinese_fonts()

def analyze_bike_data(file_path, top_n=10):
    """
    分析共享单车数据，找出热门站点和路线，并生成可视化图表。

    参数:
    file_path (str): CSV文件的路径。
    top_n (int): 覀显示的热门项目数量。
    """
    # --- 1. 数据加载 ---
    print(f"正在从 '{file_path}' 加载数据...")
    try:
        # 使用 engine='python' 增强对复杂路径或特殊字符的兼容性
        df = pd.read_csv(file_path, engine='python')
        print("数据加载成功！")
        print(f"数据总行数: {len(df)}")
        print("数据列名:", df.columns.tolist())
        print("\n前5行数据:")
        print(df.head())
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'。请检查文件路径是否正确。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # --- 2. 数据处理与分析 ---
    
    # 检查必需的列是否存在
    required_columns = ['start_station_name', 'end_station_name']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"错误：缺少以下必需列: {missing_columns}")
        print("请检查您的CSV文件格式")
        return
    
    # 检查是否有空值
    print(f"\n空值检查:")
    print(f"出发站点空值数量: {df['start_station_name'].isnull().sum()}")
    print(f"到达站点空值数量: {df['end_station_name'].isnull().sum()}")
    
    # 删除空值行
    df_clean = df.dropna(subset=['start_station_name', 'end_station_name'])
    print(f"清理空值后剩余数据行数: {len(df_clean)}")
    
    # a. 分析最热门的出发站点
    print(f"\n正在分析排名前 {top_n} 的热门出发站点...")
    top_start_stations = df_clean['start_station_name'].value_counts().nlargest(top_n)
    print("热门出发站点:")
    for i, (station, count) in enumerate(top_start_stations.items(), 1):
        print(f"  {i}. {station}: {count} 次")

    # b. 分析最热门的到达站点
    print(f"\n正在分析排名前 {top_n} 的热门到达站点...")
    top_end_stations = df_clean['end_station_name'].value_counts().nlargest(top_n)
    print("热门到达站点:")
    for i, (station, count) in enumerate(top_end_stations.items(), 1):
        print(f"  {i}. {station}: {count} 次")

    # c. 分析最热门的路线
    print(f"\n正在分析排名前 {top_n} 的热门路线...")
    # 创建一个 'route' 列来表示从起点到终点的完整路线
    df_clean['route'] = df_clean['start_station_name'] + ' -> ' + df_clean['end_station_name']
    top_routes = df_clean['route'].value_counts().nlargest(top_n)
    print("热门路线:")
    for i, (route, count) in enumerate(top_routes.items(), 1):
        print(f"  {i}. {route}: {count} 次")

    # --- 3. 可视化 ---
    print("\n正在生成可视化图表...")

    # 中文字体已经在程序开头设置
    print("中文字体设置已应用，图表中的中文应该能正常显示。")

    # 使用seaborn设置更好的样式
    sns.set_style("whitegrid")
    plt.style.use('seaborn-v0_8-darkgrid')

    # 创建一个大画布，包含3个子图
    fig, axes = plt.subplots(3, 1, figsize=(16, 20))
    fig.suptitle('共享单车热门站点与路线分析', fontsize=26, y=0.96, weight='bold', color='#2C3E50')

    # 定义更简洁的颜色方案
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']  # 简化颜色方案

    # 图1: 热门出发站点
    bars1 = axes[0].barh(range(len(top_start_stations)), top_start_stations.values, 
                     color=colors[:len(top_start_stations)], edgecolor='white', linewidth=1.2)
    axes[0].set_yticks(range(len(top_start_stations)))
    axes[0].set_yticklabels([label[:20] + '...' if len(label) > 20 else label for label in top_start_stations.index],  # 截断过长标签
                        fontsize=13, color='#2C3E50')
    axes[0].set_title(f'排名前 {top_n} 的热门出发站点', fontsize=22, weight='bold', pad=25, color='#2C3E50')
    axes[0].set_xlabel('骑行次数', fontsize=16, weight='bold', color='#2C3E50')
    axes[0].set_ylabel('站点名称', fontsize=16, weight='bold', color='#2C3E50')
    axes[0].invert_yaxis()  # 最高值在顶部
    axes[0].grid(axis='x', alpha=0.3, color='#BDC3C7')
    axes[0].set_facecolor('#F8F9F9')

    # 在条形图上添加数值标签
    for i, (bar, value) in enumerate(zip(bars1, top_start_stations.values)):
        axes[0].text(bar.get_width() + max(top_start_stations.values)*0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:,}', ha='left', va='center', fontsize=13, weight='bold', color='#2C3E50')

    # 图2: 热门到达站点
    bars2 = axes[1].barh(range(len(top_end_stations)), top_end_stations.values, 
                     color=colors[:len(top_end_stations)], edgecolor='white', linewidth=1.2)
    axes[1].set_yticks(range(len(top_end_stations)))
    axes[1].set_yticklabels([label[:20] + '...' if len(label) > 20 else label for label in top_end_stations.index],  # 截断过长标签
                        fontsize=13, color='#2C3E50')
    axes[1].set_title(f'排名前 {top_n} 的热门到达站点', fontsize=22, weight='bold', pad=25, color='#2C3E50')
    axes[1].set_xlabel('骑行次数', fontsize=16, weight='bold', color='#2C3E50')
    axes[1].set_ylabel('站点名称', fontsize=16, weight='bold', color='#2C3E50')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3, color='#BDC3C7')
    axes[1].set_facecolor('#F8F9F9')

    # 在条形图上添加数值标签
    for i, (bar, value) in enumerate(zip(bars2, top_end_stations.values)):
        axes[1].text(bar.get_width() + max(top_end_stations.values)*0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:,}', ha='left', va='center', fontsize=13, weight='bold', color='#2C3E50')

    # 图3: 热门路线
    bars3 = axes[2].barh(range(len(top_routes)), top_routes.values, 
                     color=colors[:len(top_routes)], edgecolor='white', linewidth=1.2)
    axes[2].set_yticks(range(len(top_routes)))
    axes[2].set_yticklabels([label[:20] + '...' if len(label) > 20 else label for label in top_routes.index],  # 截断过长标签
                        fontsize=12, color='#2C3E50')
    axes[2].set_title(f'排名前 {top_n} 的热门路线', fontsize=22, weight='bold', pad=25, color='#2C3E50')
    axes[2].set_xlabel('骑行次数', fontsize=16, weight='bold', color='#2C3E50')
    axes[2].set_ylabel('路线', fontsize=16, weight='bold', color='#2C3E50')
    axes[2].invert_yaxis()
    axes[2].grid(axis='x', alpha=0.3, color='#BDC3C7')
    axes[2].set_facecolor('#F8F9F9')

    # 在条形图上添加数值标签
    for i, (bar, value) in enumerate(zip(bars3, top_routes.values)):
        axes[2].text(bar.get_width() + max(top_routes.values)*0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:,}', ha='left', va='center', fontsize=13, weight='bold', color='#2C3E50')

    # 调整布局以防标签重叠
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # --- 4. 保存图表 ---
    output_filename = 'popular_stations_and_routes_elegant.png'
    # 优化图表保存设置，提高输出质量
    plt.savefig(output_filename, 
                dpi=300,  # 高分辨率
                bbox_inches='tight', 
                facecolor='white', 
                edgecolor='none',
                format='png',  # 明确指定格式
                transparent=False)  # 不使用透明背景
    print(f"分析完成！图表已保存为 '{output_filename}'。")

    # 显示图表（如果您在本地环境中运行，会弹出一个窗口）
    plt.show()

    # --- 5. 输出统计信息 ---
    print(f"\n=== 数据统计摘要 ===")
    print(f"总骑行次数: {len(df_clean)}")
    print(f"不同出发站点数: {df_clean['start_station_name'].nunique()}")
    print(f"不同到达站点数: {df_clean['end_station_name'].nunique()}")
    print(f"不同路线数: {df_clean['route'].nunique()}")


if __name__ == '__main__':
    # 请将这里的路径替换为您自己文件的实际路径
    file_path = r'E:\fire_flower\2025年度“火花杯”数学建模精英联赛-B题-附件\202503-capitalbikeshare-tripdata.csv'
    
    # 检查文件是否存在
    if os.path.exists(file_path):
        analyze_bike_data(file_path, top_n=10)
    else:
        print(f"错误：文件路径 '{file_path}' 不存在。")
        print("请打开脚本文件，并将变量 'file_path' 的值修改为您的CSV文件的正确路径。")