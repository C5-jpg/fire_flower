# -*- coding: utf-8 -*-
# 共享单车数据分析脚本 (无复杂注释版)
# 目标: 分析 demand_features.csv 和 202503-capitalbikeshare-tripdata.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform
import numpy as np

def setup_chinese_fonts():
    system = platform.system()
    if system == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    elif system == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_fonts()
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# 正确处理包含中文引号的路径 (直接使用原始中文引号字符)
DATA_DIR = 'C:\\Users\\hk\\fire_flower\\2025年度“火花杯”数学建模精英联赛-B题-附件'
# 调试：打印实际使用的 DATA_DIR 路径
print(f"DEBUG: DATA_DIR is set to: {DATA_DIR}")
DEMAND_FEATURES_FILE = os.path.join(DATA_DIR, 'demand_features.csv')
TRIPDATA_FILE = os.path.join(DATA_DIR, '202503-capitalbikeshare-tripdata.csv')

# 图表输出文件名
OUTPUT_FILES = {
    'demand_corr_heatmap': 'demand_features_correlation_heatmap.png',
    'popular_stations_and_routes': 'popular_stations_and_routes_final_no_comments.png'
}

TOP_N = 10
CHUNKSIZE = 50000

def load_demand_features(file_path):
    print(f"[1/3] Loading '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
        print("demand_features.csv loaded successfully!")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading demand_features.csv: {e}")
        return None

def analyze_demand_features(df):
    print("\n[2/3] Analyzing demand_features.csv ...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric columns: {numeric_cols}")
    
    if not numeric_cols:
        print("No numeric columns found. Skipping analysis.")
        return
    
    print("  -> Generating correlation heatmap...")
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Feature Correlation Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILES['demand_corr_heatmap'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     Chart saved to: {OUTPUT_FILES['demand_corr_heatmap']}")
    print("  -> Analysis complete.\n")


def analyze_tripdata_in_chunks(file_path, top_n=TOP_N, chunksize=CHUNKSIZE):
    print(f"[3/3] Analyzing '{file_path}' in chunks...")
    print(f"Chunk size: {chunksize}")
    
    start_counts = {}
    end_counts = {}
    route_counts = {}
    total_rows = 0
    chunk_count = 0
    
    try:
        chunk_iter = pd.read_csv(file_path, chunksize=chunksize, engine='python')
        for chunk in chunk_iter:
            chunk_count += 1
            total_rows += len(chunk)
            if chunk_count % 10 == 0 or chunk_count == 1:
                print(f"  Processed chunk {chunk_count}, total rows: {total_rows}")
            
            required_cols = ['start_station_name', 'end_station_name']
            missing_cols = [col for col in required_cols if col not in chunk.columns]
            if missing_cols:
                print(f"    Warning: Missing columns {missing_cols} in this chunk. Skipping.")
                continue
            
            chunk_clean = chunk.dropna(subset=required_cols)
            
            # Count start stations
            start_series = chunk_clean['start_station_name'].value_counts()
            for station, count in start_series.items():
                start_counts[station] = start_counts.get(station, 0) + count

            # Count end stations
            end_series = chunk_clean['end_station_name'].value_counts()
            for station, count in end_series.items():
                end_counts[station] = end_counts.get(station, 0) + count

            # Count routes
            chunk_clean['route'] = chunk_clean['start_station_name'] + ' -> ' + chunk_clean['end_station_name']
            route_series = chunk_clean['route'].value_counts()
            for route, count in route_series.items():
                route_counts[route] = route_counts.get(route, 0) + count

        print(f"Analysis complete. Total rows processed: {total_rows}")
        
        top_start = pd.Series(start_counts).nlargest(top_n)
        top_end = pd.Series(end_counts).nlargest(top_n)
        top_routes = pd.Series(route_counts).nlargest(top_n)
        
        return top_start, top_end, top_routes
        
    except Exception as e:
        print(f"Error analyzing tripdata: {e}")
        return None, None, None

def visualize_popular_items(top_start, top_end, top_routes, top_n=TOP_N):
    if top_start is None or top_end is None or top_routes is None:
        print("Data missing, cannot generate visualization.")
        return
        
    print("\nGenerating visualization for popular items...")
    colors = sns.color_palette("husl", top_n)
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    fig.suptitle('Popular Stations and Routes', fontsize=20)

    # Top start stations
    bars1 = axes[0].barh(range(len(top_start)), top_start.values, color=colors)
    axes[0].set_yticks(range(len(top_start)))
    y_labels_start = [label[:20] + '...' if len(label) > 20 else label for label in top_start.index]
    axes[0].set_yticklabels(y_labels_start)
    axes[0].set_title(f'Top {top_n} Start Stations')
    axes[0].set_xlabel('Count')
    axes[0].invert_yaxis()
    for i, (bar, value) in enumerate(zip(bars1, top_start.values)):
        axes[0].text(bar.get_width() + max(top_start.values)*0.01, bar.get_y() + bar.get_height()/2, 
                     f'{value:,}', va='center')

    # Top end stations
    bars2 = axes[1].barh(range(len(top_end)), top_end.values, color=colors)
    axes[1].set_yticks(range(len(top_end)))
    y_labels_end = [label[:20] + '...' if len(label) > 20 else label for label in top_end.index]
    axes[1].set_yticklabels(y_labels_end)
    axes[1].set_title(f'Top {top_n} End Stations')
    axes[1].set_xlabel('Count')
    axes[1].invert_yaxis()
    for i, (bar, value) in enumerate(zip(bars2, top_end.values)):
        axes[1].text(bar.get_width() + max(top_end.values)*0.01, bar.get_y() + bar.get_height()/2, 
                     f'{value:,}', va='center')

    # Top routes
    bars3 = axes[2].barh(range(len(top_routes)), top_routes.values, color=colors)
    axes[2].set_yticks(range(len(top_routes)))
    y_labels_route = [label[:30] + '...' if len(label) > 30 else label for label in top_routes.index]
    axes[2].set_yticklabels(y_labels_route)
    axes[2].set_title(f'Top {top_n} Routes')
    axes[2].set_xlabel('Count')
    axes[2].invert_yaxis()
    for i, (bar, value) in enumerate(zip(bars3, top_routes.values)):
        axes[2].text(bar.get_width() + max(top_routes.values)*0.01, bar.get_y() + bar.get_height()/2, 
                     f'{value:,}', va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = OUTPUT_FILES['popular_stations_and_routes']
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {output_file}")
    print("--- Visualization complete ---\n")

if __name__ == "__main__":
    print("=========================================")
    print("=== Bike Data Analysis Script (No Comments) ===")
    print("=========================================")
    
    if not os.path.exists(DEMAND_FEATURES_FILE):
        print(f"Error: File not found {DEMAND_FEATURES_FILE}")
        exit(1)
    if not os.path.exists(TRIPDATA_FILE):
        print(f"Error: File not found {TRIPDATA_FILE}")
        exit(1)

    # 1. Analyze demand_features.csv
    df_demand = load_demand_features(DEMAND_FEATURES_FILE)
    if df_demand is not None:
        analyze_demand_features(df_demand)
    
    # 2. Analyze tripdata.csv (chunked)
    top_start, top_end, top_routes = analyze_tripdata_in_chunks(TRIPDATA_FILE)
    
    # 3. Visualize
    visualize_popular_items(top_start, top_end, top_routes)
    
    print("=========================================")
    print("=== All tasks completed ===")
    print("=========================================")