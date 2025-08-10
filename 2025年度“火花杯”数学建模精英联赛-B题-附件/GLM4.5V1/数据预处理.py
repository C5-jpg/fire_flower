import pandas as pd
import numpy as np
import os
from datetime import datetime

# 加载数据
def load_data(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = []
    
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
    
    # 合并所有数据
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

# 数据清洗
def clean_data(df):
    # 处理缺失值
    df = df.dropna(subset=['started_at', 'ended_at', 'start_lat', 'start_lng', 'end_lat', 'end_lng'])
    
    # 转换时间格式
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    
    # 计算骑行时长(分钟)
    df['duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    
    # 过滤异常值(例如骑行时间过长或过短)
    df = df[(df['duration'] > 1) & (df['duration'] < 180)]
    
    # 提取时间特征
    df['hour'] = df['started_at'].dt.hour
    df['day_of_week'] = df['started_at'].dt.dayofweek
    df['month'] = df['started_at'].dt.month
    
    return df


# 主函数
if __name__ == '__main__':
    # 文件路径
    file_path = r'C:\Users\hk\fire_flower\2025年度“火花杯”数学建模精英联赛-B题-附件\202503-capitalbikeshare-tripdata.csv'
    
    # 加载单个文件
    df = pd.read_csv(file_path)
    print(f'原始数据形状: {df.shape}')
    
    # 数据清洗
    cleaned_df = clean_data(df)
    print(f'清洗后数据形状: {cleaned_df.shape}')
    
    # 打印前几行数据
    print('\n清洗后的数据样例:')
    print(cleaned_df.head())
    
    # 可以根据需要添加更多分析或保存结果的代码
    # cleaned_df.to_csv('cleaned_data.csv', index=False)



    