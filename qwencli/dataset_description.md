
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
