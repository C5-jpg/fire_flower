# 预测需求数据说明

为了完成需求预测任务，我们需要构建一个包含以下关键特征的数据集：

## 核心特征

1.  **日期时间信息 (Date/Time Features):**
    *   `date`: 具体日期 (YYYY-MM-DD)。
    *   `day_of_week`: 星期几 (0-6 或 Monday-Sunday)。
    *   `month`: 月份 (1-12)。
    *   `hour`: 小时 (0-23)，如果预测粒度是小时级别。
    *   `is_weekend`: 是否为周末 (布尔值)。

2.  **POI 得分 (POI Score):**
    *   `poi_score`: 一个反映该区域或地点吸引力的数值。这个分数可以基于POI（兴趣点）的类型、数量、密度、评价等综合计算得出。例如，商业区、交通枢纽、旅游景点的POI得分通常较高。

3.  **节假日系数 (Holiday Factor):**
    *   `is_holiday`: 是否为法定节假日 (布尔值)。
    *   `holiday_name`: 节假日名称 (字符串)，有助于区分不同类型的假期（如春节、国庆节、圣诞节）对需求的影响可能不同。
    *   `days_to_nearest_holiday`: 距离最近节假日的天数 (整数)，可以捕捉节假日前后的效应。
    *   `holiday_factor`: 一个数值化的节假日影响系数，可以直接量化节假日对需求的放大或抑制作用。

4.  **用户类型比例 (User Type Proportion):**
    *   `resident_ratio`: 常驻居民用户比例。
    *   `visitor_ratio`: 游客/临时访客用户比例。
    *   `commuter_ratio`: 通勤用户比例。
    *   (注意：这些比例之和应为1或100%)。不同类型的用户在不同时间、地点的需求模式可能差异很大。

5.  **天气数据 (Weather Data):**
    *   `temperature`: 温度。
    *   `weather_condition`: 天气状况 (晴天、雨天、雪天等)。
    *   `humidity`: 湿度。
    *   `wind_speed`: 风速。
    *   天气对出行需求有显著影响。

6.  **历史需求数据 (Historical Demand Data):**
    *   `historical_demand_lag_1`: 前一天（或前一小时）的同一时间点的需求量。
    *   `historical_demand_lag_7`: 一周前的同一时间点的需求量。
    *   `historical_demand_rolling_mean_7`: 过去7天的平均需求量。
    *   历史数据是预测未来需求最直接和重要的依据。

## 目标变量

*   `demand`: 在特定时间、地点的实际需求量（例如，订单数、客流量、车辆使用次数等）。这是模型需要预测的值。

## 数据示例 (结构化格式)

| date       | day_of_week | month | hour | poi_score | is_holiday | holiday_name | days_to_nearest_holiday | holiday_factor | resident_ratio | visitor_ratio | commuter_ratio | temperature | weather_condition | historical_demand_lag_1 | demand |
| :--------- | :---------- | :---- | :--- | :-------- | :--------- | :----------- | :---------------------- | :------------- | :------------- | :------------ | :------------- | :---------- | :---------------- | :---------------------- | :----- |
| 2023-10-01 | 6 (Sunday)  | 10    | 12   | 0.85      | True       | 国庆节        | 0                       | 1.5            | 0.3            | 0.6           | 0.1            | 22          | 晴天              | 150                     | 225    |
| 2023-10-02 | 0 (Monday)  | 10    | 8    | 0.70      | False      |              | 1                       | 1.0            | 0.5            | 0.2           | 0.3            | 18          | 多云              | 80                      | 100    |
| 2023-10-02 | 0 (Monday)  | 10    | 18   | 0.90      | False      |              | 1                       | 1.0            | 0.4            | 0.1           | 0.5            | 18          | 多云              | 200                     | 250    |

通过收集和整合包含以上特征的数据集，就可以利用机器学习模型（如线性回归、随机森林、XGBoost、LSTM等）来训练和预测需求。