import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. 生成基础日期时间序列 ---
start_date = '2023-01-01'
end_date = '2023-12-31'
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# --- 2. 定义节假日信息 ---
# 这里以2023年中国部分节假日为例
holidays = {
    '2023-01-01': '元旦',
    '2023-01-22': '春节', '2023-01-23': '春节', '2023-01-24': '春节', '2023-01-25': '春节',
    '2023-01-26': '春节', '2023-01-27': '春节', '2023-01-28': '春节',
    '2023-04-05': '清明节',
    '2023-04-29': '劳动节', '2023-04-30': '劳动节', '2023-05-01': '劳动节', '2023-05-02': '劳动节',
    '2023-05-03': '劳动节',
    '2023-06-22': '端午节', '2023-06-23': '端午节', '2023-06-24': '端午节',
    '2023-09-29': '中秋节', '2023-09-30': '中秋节',
    '2023-10-01': '国庆节', '2023-10-02': '国庆节', '2023-10-03': '国庆节', '2023-10-04': '国庆节',
    '2023-10-05': '国庆节', '2023-10-06': '国庆节',
}

# --- 3. 创建数据列表 ---
data = []
# 假设我们有3个不同的区域/POI
poi_ids = ['POI_A', 'POI_B', 'POI_C']
poi_base_scores = {'POI_A': 0.9, 'POI_B': 0.6, 'POI_C': 0.3} # 基础POI得分

for date in dates:
    for poi_id in poi_ids:
        entry = {}
        entry['date'] = date
        entry['poi_id'] = poi_id
        entry['day_of_week'] = date.weekday()
        entry['month'] = date.month
        entry['is_weekend'] = date.weekday() >= 5
        
        # --- 4. 计算节假日相关特征 ---
        date_str = date.strftime('%Y-%m-%d')
        if date_str in holidays:
            entry['is_holiday'] = True
            entry['holiday_name'] = holidays[date_str]
            entry['days_to_nearest_holiday'] = 0
        else:
            entry['is_holiday'] = False
            entry['holiday_name'] = ''
            # 计算距离最近节假日的天数
            nearest_holiday_diff = min([abs((pd.to_datetime(holiday_date) - date).days) for holiday_date in holidays.keys()])
            entry['days_to_nearest_holiday'] = nearest_holiday_diff
            
        # --- 5. 简化的节假日系数 ---
        # 这里用一个简单的逻辑，实际应用中可能更复杂
        if entry['is_holiday']:
            entry['holiday_factor'] = 1.5
        elif entry['days_to_nearest_holiday'] <= 1: # 节假日前后一天
            entry['holiday_factor'] = 1.2
        else:
            entry['holiday_factor'] = 1.0
            
        # --- 6. POI得分 ---
        # 基础得分 + 一些随机噪声来模拟真实情况
        base_score = poi_base_scores[poi_id]
        noise = np.random.normal(0, 0.05)
        entry['poi_score'] = np.clip(base_score + noise, 0, 1) # 限制在0-1之间
            
        # --- 7. 用户类型比例 ---
        # 这里为每个POI设定一个基础比例，并加入一些随机波动
        if poi_id == 'POI_A': # 假设POI_A是商业区，游客多
            base_resident, base_visitor, base_commuter = 0.2, 0.6, 0.2
        elif poi_id == 'POI_B': # 假设POI_B是住宅区，居民多
            base_resident, base_visitor, base_commuter = 0.7, 0.1, 0.2
        else: # 假设POI_C是交通枢纽，通勤者多
            base_resident, base_visitor, base_commuter = 0.3, 0.2, 0.5
            
        # 添加随机波动并归一化
        ratios = np.array([base_resident, base_visitor, base_commuter]) + np.random.normal(0, 0.05, size=3)
        # 确保比例非负
        ratios = np.maximum(ratios, 0)
        # 归一化
        ratios = ratios / ratios.sum()
        
        entry['resident_ratio'] = ratios[0]
        entry['visitor_ratio'] = ratios[1]
        entry['commuter_ratio'] = ratios[2]
        
        # --- 8. 天气数据 (简化模拟) ---
        # 这里用一个非常简化的模型生成天气数据
        # 温度随季节变化
        day_of_year = date.timetuple().tm_yday
        avg_temp = 20 * np.cos(2 * np.pi * (day_of_year - 10) / 365) + 15 # 简化的年度温度变化模型
        temp_noise = np.random.normal(0, 5)
        entry['temperature'] = avg_temp + temp_noise
        
        # 天气状况 (简化)
        weather_types = ['晴天', '多云', '小雨', '大雨']
        # 晴天概率最高，雨天概率随温度降低而增加（不准确但为示例）
        # 确保概率非负
        rain_prob = max(0.0, 0.1 + 0.2 * (1 - (entry['temperature'] + 10) / 40)) # 粗略模拟
        weights = [max(0.0, 0.6 - rain_prob/2), max(0.0, 0.3 - rain_prob/2), max(0.0, rain_prob * 0.7), max(0.0, rain_prob * 0.3)]
        # 确保权重总和大于0，避免除零错误
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            entry['weather_condition'] = np.random.choice(weather_types, p=normalized_weights)
        else:
            # 如果所有权重都是0，则均匀分布选择
            entry['weather_condition'] = np.random.choice(weather_types)
        
        entry['humidity'] = np.random.uniform(30, 90)
        entry['wind_speed'] = np.random.uniform(0, 20)
        
        data.append(entry)

# --- 9. 创建DataFrame ---
df = pd.DataFrame(data)

# --- 10. 生成历史需求和目标变量 ---
# 这是一个非常简化的模拟过程，实际中会基于真实历史数据
df = df.sort_values(by=['poi_id', 'date']).reset_index(drop=True)

# 初始化历史需求列
df['historical_demand_lag_1'] = np.nan
df['historical_demand_lag_7'] = np.nan
df['historical_demand_rolling_mean_7'] = np.nan

# 初始化目标变量
df['demand'] = np.nan

# 为每个POI分别生成
for poi_id in poi_ids:
    poi_mask = df['poi_id'] == poi_id
    poi_df = df[poi_mask].copy()
    
    for i in range(len(poi_df)):
        date = poi_df.iloc[i]['date']
        # 基础需求由POI得分、用户比例、节假日、天气等因素综合决定
        base_demand = (
            poi_df.iloc[i]['poi_score'] * 100 +
            poi_df.iloc[i]['visitor_ratio'] * 200 + 
            poi_df.iloc[i]['commuter_ratio'] * 150 +
            (1 if poi_df.iloc[i]['is_weekend'] else 0) * 50 +
            (poi_df.iloc[i]['holiday_factor'] - 1) * 100
        )
        
        # 天气影响
        weather_impact = 0
        if poi_df.iloc[i]['weather_condition'] == '小雨':
            weather_impact = -20
        elif poi_df.iloc[i]['weather_condition'] == '大雨':
            weather_impact = -50
        # 温度影响 (简化)
        temp = poi_df.iloc[i]['temperature']
        if temp < 0 or temp > 35:
            weather_impact -= 30
        elif 0 <= temp < 10 or 25 <= temp < 35:
            weather_impact -= 10
            
        base_demand += weather_impact
        
        # 添加随机噪声
        noise = np.random.normal(0, 20)
        current_demand = max(0, int(base_demand + noise)) # 需求不能为负
        
        # 设置历史需求特征
        if i >= 1:
            df.loc[poi_df.index[i], 'historical_demand_lag_1'] = df.loc[poi_df.index[i-1], 'demand']
        if i >= 7:
            df.loc[poi_df.index[i], 'historical_demand_lag_7'] = df.loc[poi_df.index[i-7], 'demand']
            # 计算滚动平均 (使用已计算出的需求)
            recent_demands = [df.loc[poi_df.index[i-j], 'demand'] for j in range(1, 8)]
            df.loc[poi_df.index[i], 'historical_demand_rolling_mean_7'] = np.mean(recent_demands)
        elif i > 0: # 对于前7天但大于1天的数据，可以用可用天数计算平均
             recent_demands = [df.loc[poi_df.index[i-j], 'demand'] for j in range(1, min(i+1, 8))]
             df.loc[poi_df.index[i], 'historical_demand_rolling_mean_7'] = np.mean(recent_demands) if recent_demands else np.nan
        
        # 设置目标变量
        df.loc[poi_df.index[i], 'demand'] = current_demand

# --- 11. 保存数据 ---
# 保存完整的数据集
df.to_csv('demand_forecast_data.csv', index=False, encoding='utf-8-sig')

# 保存数据描述
description = """
# 需求预测数据集

本数据集用于模拟一个基于地点（POI）的需求预测任务。数据涵盖了2023年一整年，包含3个不同的地点（POI_A, POI_B, POI_C）。

## 字段说明

- `date`: 日期 (YYYY-MM-DD)
- `poi_id`: 地点/POI标识符
- `day_of_week`: 星期几 (0=Monday, 6=Sunday)
- `month`: 月份
- `is_weekend`: 是否为周末
- `is_holiday`: 是否为节假日
- `holiday_name`: 节假日名称
- `days_to_nearest_holiday`: 距离最近节假日的天数
- `holiday_factor`: 节假日系数
- `poi_score`: POI得分 (0-1)
- `resident_ratio`: 居民用户比例
- `visitor_ratio`: 游客用户比例
- `commuter_ratio`: 通勤用户比例
- `temperature`: 温度 (摄氏度)
- `weather_condition`: 天气状况
- `humidity`: 湿度 (%)
- `wind_speed`: 风速 (km/h)
- `historical_demand_lag_1`: 前一天的需求量
- `historical_demand_lag_7`: 7天前的需求量
- `historical_demand_rolling_mean_7`: 过去7天的平均需求量
- `demand`: 当天的实际需求量 (目标变量)

数据已生成并保存为 'demand_forecast_data.csv'。
"""
with open('dataset_description.md', 'w', encoding='utf-8') as f:
    f.write(description)

print("数据集 'demand_forecast_data.csv' 和描述文件 'dataset_description.md' 已生成。")