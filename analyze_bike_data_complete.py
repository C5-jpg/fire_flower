# -*- coding: utf-8 -*-
\"\"\"
共享单车数据分析脚本 (完整版 - 使用正确路径)

此脚本旨在分析 '2025年度“火花杯”数学建模精英联赛-B题-附件' 目录下的两个CSV文件:
1. demand_features.csv (较小)
2. 202503-capitalbikeshare-tripdata.csv (较大, 需分块处理)

功能包括:
- 加载并探索数据结构
- 对demand_features.csv进行相关性分析和可视化
- 对202503-capitalbikeshare-tripdata.csv进行热门站点和路线分析 (使用分块读取)
- 生成可视化图表
\"\"\"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform
import numpy as np

# --- 配置 ---
# 设置中文字体和解决负号显示问题
def setup_chinese_fonts():
    \"\"\"根据操作系统设置合适的中文字体\"\"\"
    system = platform.system()
    if system == \"Windows\":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    elif system == \"Darwin\": # macOS
        plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Arial Unicode MS']
    else: # Linux or other
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_fonts()

# Seaborn样式
sns.set_style(\"whitegrid\")
plt.style.use('seaborn-v0_8-darkgrid')

# 文件路径 (使用正确的Unicode转义)
DATA_DIR = 'C:\\\\Users\\\\hk\\\\fire_flower\\\\2025年度\u201c火花杯\u201d数学建模精英联赛-B题-附件'
DEMAND_FEATURES_FILE = os.path.join(DATA_DIR, 'demand_features.csv')
TRIPDATA_FILE = os.path.join(DATA_DIR, '202503-capitalbikeshare-tripdata.csv')

# 检查文件是否存在
if not os.path.exists(DEMAND_FEATURES_FILE):
    print(f\"错误: 找不到文件 {DEMAND_FEATURES_FILE}\")
    exit(1)

if not os.path.exists(TRIPDATA_FILE):
    print(f\"错误: 找不到文件 {TRIPDATA_FILE}\")
    exit(1)

print(f\"确认文件存在:\")
print(f\"  - {DEMAND_FEATURES_FILE}\")
print(f\"  - {TRIPDATA_FILE}\")


# 输出图表文件名
OUTPUT_FILES = {
    'demand_corr_heatmap': 'demand_features_correlation_heatmap.png',
    'demand_scatter_matrix': 'demand_features_scatter_matrix.png',
    'demand_histograms': 'demand_features_histograms.png',
    'popular_stations_and_routes': 'popular_stations_and_routes_final.png'
}

# 分析参数
TOP_N = 10
CHUNKSIZE = 50000 # 分块读取的块大小

# --- 数据加载与分析函数 ---

def load_demand_features(file_path):
    \"\"\"
    加载并初步分析 demand_features.csv
    \"\"\"
    print(f\"\\n[1/3] 正在加载 '{file_path}'...\")
    try:
        df = pd.read_csv(file_path)
        print(\"demand_features.csv 加载成功!\")
        print(f\"数据形状: {df.shape}\")
        print(\"\\n列名:\")
        print(df.columns.tolist())
        print(\"\\n数据类型:\")
        print(df.dtypes)
        print(\"\\n前5行数据:\")
        print(df.head())
        print(\"\\n基本统计信息:\")
        print(df.describe())
        return df
    except Exception as e:
        print(f\"加载 demand_features.csv 时出错: {e}\")
        return None

def analyze_demand_features(df):
    \"\"\"
    分析 demand_features.csv 的相关性和分布
    \"\"\"
    print(\"\\n[2/3] 开始分析 demand_features.csv ...\")
    # 选择数值型列进行分析
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f\"用于分析的数值列: {numeric_cols}\")
    
    if not numeric_cols:
        print(\"没有找到数值型列，跳过相关性分析和可视化。\")
        return
    
    # 相关性热力图
    print(\"  -> 正在生成相关性热力图...\")
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=\".2f\", cmap='coolwarm', square=True,
                linewidths=.5, cbar_kws={\"shrink\": .8})
    plt.title('站点特征与需求相关性热力图', fontsize=16, weight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILES['demand_corr_heatmap'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f\"     图表已保存至: {OUTPUT_FILES['demand_corr_heatmap']}\")
    
    # 散点图矩阵 (选取几个关键特征)
    key_features = ['borrow_count', 'return_count', 'casual_ratio', 'POI_score', 'predicted_demand']
    key_features = [col for col in key_features if col in numeric_cols]
    if len(key_features) > 1:
        print(\"  -> 正在生成散点图矩阵...\")
        # 为了性能，只对部分数据进行采样
        sample_size = min(1000, len(df))
        if sample_size < len(df):
            print(f\"     (为提高性能，从 {len(df)} 行中随机采样 {sample_size} 行进行绘图)\")
        pair_plot_df = df[key_features].sample(n=sample_size, random_state=42) 
        g = sns.pairplot(pair_plot_df, diag_kind='kde', plot_kws={'alpha': 0.6})
        g.fig.suptitle('站点特征散点图矩阵', y=1.02, fontsize=16, weight='bold')
        plt.savefig(OUTPUT_FILES['demand_scatter_matrix'], dpi=300, bbox_inches='tight')
        plt.close()
        print(f\"     图表已保存至: {OUTPUT_FILES['demand_scatter_matrix']}\")
    else:
        print(\"     可用于散点图的特征少于2个，跳过散点图矩阵。\")
    
    # 特征分布直方图
    print(\"  -> 正在生成特征分布直方图...\")
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], bins=30, color=sns.color_palette(\"husl\", len(numeric_cols))[i])
        axes[i].set_title(f'{col} 分布', fontsize=12, weight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('频率')
        
    # 隐藏多余的子图
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle('站点特征分布直方图', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_FILES['demand_histograms'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f\"     图表已保存至: {OUTPUT_FILES['demand_histograms']}\")
    
    print(\"  -> demand_features.csv 分析完成。\\n\")

def analyze_tripdata_in_chunks(file_path, top_n=TOP_N, chunksize=CHUNKSIZE):
    \"\"\"
    分块读取并分析 202503-capitalbikeshare-tripdata.csv
    统计热门站点和路线
    \"\"\"
    print(f\"[3/3] 开始分块读取并分析 '{file_path}'...\")
    print(f\"分块大小: {chunksize}\")
    
    # 初始化计数器
    start_station_counts = {}
    end_station_counts = {}
    route_counts = {}
    
    total_rows = 0
    chunk_count = 0
    
    try:
        # 使用分块迭代器
        chunk_iter = pd.read_csv(file_path, chunksize=chunksize, engine='python')
        
        for chunk in chunk_iter:
            chunk_count += 1
            total_rows += len(chunk)
            if chunk_count % 10 == 0 or chunk_count == 1:
                print(f\"  已处理块 {chunk_count}, 当前总行数: {total_rows}\")
            
            # 检查必要的列是否存在
            required_cols = ['start_station_name', 'end_station_name']
            missing_cols = [col for col in required_cols if col not in chunk.columns]
            if missing_cols:
                print(f\"    警告: 当前块缺少列 {missing_cols}，跳过该块的站点/路线统计。\")
                continue
            
            # 删除这两列的空值行
            chunk_clean = chunk.dropna(subset=required_cols)
            
            # 统计出发站点
            start_counts = chunk_clean['start_station_name'].value_counts()
            for station, count in start_counts.items():
                start_station_counts[station] = start_station_counts.get(station, 0) + count
                    
            # 统计到达站点
            end_counts = chunk_clean['end_station_name'].value_counts()
            for station, count in end_counts.items():
                end_station_counts[station] = end_station_counts.get(station, 0) + count
                    
            # 统计路线
            chunk_clean['route'] = chunk_clean['start_station_name'] + ' -> ' + chunk_clean['end_station_name']
            route_counts_series = chunk_clean['route'].value_counts()
            for route, count in route_counts_series.items():
                route_counts[route] = route_counts.get(route, 0) + count
                    
        print(f\"文件读取和统计分析完成。总处理行数: {total_rows}\")
        
        # 转换为Series并排序
        top_start_stations = pd.Series(start_station_counts).nlargest(top_n)
        top_end_stations = pd.Series(end_station_counts).nlargest(top_n)
        top_routes = pd.Series(route_counts).nlargest(top_n)
        
        print(f\"\\n--- 热门出发站点 Top {top_n} ---\")
        print(top_start_stations)
        print(f\"\\n--- 热门到达站点 Top {top_n} ---\")
        print(top_end_stations)
        print(f\"\\n--- 热门路线 Top {top_n} ---\")
        print(top_routes)
        
        return top_start_stations, top_end_stations, top_routes
        
    except Exception as e:
        print(f\"分析 tripdata 时出错: {e}\")
        return None, None, None


def visualize_popular_items(top_start_stations, top_end_stations, top_routes, top_n=TOP_N):
    \"\"\"
    可视化热门站点和路线
    \"\"\"
    if top_start_stations is None or top_end_stations is None or top_routes is None:
        print(\"缺少数据，无法生成可视化图表。\")
        return
        
    print(\"\\n正在生成热门站点与路线可视化图表...\")
    
    # 定义颜色方案
    colors = sns.color_palette(\"husl\", top_n)
    
    # 创建一个大画布，包含3个子图
    fig, axes = plt.subplots(3, 1, figsize=(16, 22))
    fig.suptitle('共享单车热门站点与路线分析 (完整版)', fontsize=26, y=0.96, weight='bold', color='#2C3E50')

    # 图1: 热门出发站点
    bars1 = axes[0].barh(range(len(top_start_stations)), top_start_stations.values, 
                         color=colors, edgecolor='white', linewidth=1.2)
    axes[0].set_yticks(range(len(top_start_stations)))
    # 截断过长的标签
    y_labels_start = [label[:25] + '...' if len(label) > 25 else label for label in top_start_stations.index] 
    axes[0].set_yticklabels(y_labels_start, fontsize=13, color='#2C3E50')
    axes[0].set_title(f'排名前 {top_n} 的热门出发站点', fontsize=22, weight='bold', pad=25, color='#2C3E50')
    axes[0].set_xlabel('骑行次数', fontsize=16, weight='bold', color='#2C3E50')
    axes[0].set_ylabel('站点名称', fontsize=16, weight='bold', color='#2C3E50')
    axes[0].invert_yaxis()  # 最高值在顶部
    axes[0].grid(axis='x', alpha=0.3, color='#BDC3C7')
    axes[0].set_facecolor('#F8F9F9')

    # 在条形图上添加数值标签
    max_val_start = top_start_stations.max() if len(top_start_stations) > 0 else 0
    for i, (bar, value) in enumerate(zip(bars1, top_start_stations.values)):
        axes[0].text(bar.get_width() + max_val_start*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:,}', ha='left', va='center', fontsize=13, weight='bold', color='#2C3E50')

    # 图2: 热门到达站点
    bars2 = axes[1].barh(range(len(top_end_stations)), top_end_stations.values, 
                         color=colors, edgecolor='white', linewidth=1.2)
    axes[1].set_yticks(range(len(top_end_stations)))
    # 截断过长的标签
    y_labels_end = [label[:25] + '...' if len(label) > 25 else label for label in top_end_stations.index]
    axes[1].set_yticklabels(y_labels_end, fontsize=13, color='#2C3E50')
    axes[1].set_title(f'排名前 {top_n} 的热门到达站点', fontsize=22, weight='bold', pad=25, color='#2C3E50')
    axes[1].set_xlabel('骑行次数', fontsize=16, weight='bold', color='#2C3E50')
    axes[1].set_ylabel('站点名称', fontsize=16, weight='bold', color='#2C3E50')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3, color='#BDC3C7')
    axes[1].set_facecolor('#F8F9F9')

    # 在条形图上添加数值标签
    max_val_end = top_end_stations.max() if len(top_end_stations) > 0 else 0
    for i, (bar, value) in enumerate(zip(bars2, top_end_stations.values)):
        axes[1].text(bar.get_width() + max_val_end*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:,}', ha='left', va='center', fontsize=13, weight='bold', color='#2C3E50')

    # 图3: 热门路线
    bars3 = axes[2].barh(range(len(top_routes)), top_routes.values, 
                         color=colors, edgecolor='white', linewidth=1.2)
    axes[2].set_yticks(range(len(top_routes)))
    # 截断过长的标签
    y_labels_route = [label[:35] + '...' if len(label) > 35 else label for label in top_routes.index]
    axes[2].set_yticklabels(y_labels_route, fontsize=12, color='#2C3E50')
    axes[2].set_title(f'排名前 {top_n} 的热门路线', fontsize=22, weight='bold', pad=25, color='#2C3E50')
    axes[2].set_xlabel('骑行次数', fontsize=16, weight='bold', color='#2C3E50')
    axes[2].set_ylabel('路线', fontsize=16, weight='bold', color='#2C3E50')
    axes[2].invert_yaxis()
    axes[2].grid(axis='x', alpha=0.3, color='#BDC3C7')
    axes[2].set_facecolor('#F8F9F9')

    # 在条形图上添加数值标签
    max_val_route = top_routes.max() if len(top_routes) > 0 else 0
    for i, (bar, value) in enumerate(zip(bars3, top_routes.values)):
        axes[2].text(bar.get_width() + max_val_route*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:,}', ha='left', va='center', fontsize=13, weight='bold', color='#2C3E50')

    # 调整布局以防标签重叠
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # 保存图表
    output_file = OUTPUT_FILES['popular_stations_and_routes']
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='png', transparent=False)
    plt.close()
    print(f\"图表已保存至: {output_file}\")
    print(\"--- 可视化图表生成完成 ---\\n\")


# --- 主执行逻辑 ---
if __name__ == \"__main__\":
    print(\"=========================================\")
    print(\"=== 共享单车数据分析脚本 (完整版) ===\")
    print(\"=========================================\")
    
    # 1. 分析 demand_features.csv
    df_demand = load_demand_features(DEMAND_FEATURES_FILE)
    if df_demand is not None:
        analyze_demand_features(df_demand)
    
    # 2. 分析 202503-capitalbikeshare-tripdata.csv (分块处理)
    top_start, top_end, top_routes = analyze_tripdata_in_chunks(TRIPDATA_FILE)
    
    # 3. 可视化热门站点和路线
    visualize_popular_items(top_start, top_end, top_routes)
    
    print(\"=========================================\")
    print(\"=== 所有分析任务完成 ===\")
    print(\"=========================================\")