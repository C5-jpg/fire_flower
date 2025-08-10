import pandas as pd
import numpy as np

# --- 1. 生成模拟的站点和需求数据 ---

# 假设有100个站点
num_stations = 100

# 生成站点ID
station_ids = [f'S{i:03d}' for i in range(1, num_stations + 1)]

# 生成模拟的预测净需求 (借车为正，还车为负)
np.random.seed(42) # 为了结果可复现
# 假设需求服从正态分布，均值在-50到50之间，标准差为20
predicted_demand = np.random.normal(loc=np.random.uniform(-50, 50, num_stations), scale=20, size=num_stations)
# 为了更真实，可以将需求取整
predicted_demand = np.round(predicted_demand)

# 生成站点的初始容量或参考容量 (用于模拟数据，非优化变量)
# 假设容量与需求的绝对值有一定关系
base_capacity = np.abs(predicted_demand) + np.random.poisson(10, num_stations)

# 创建一个DataFrame来存储站点数据
station_data = pd.DataFrame({
    'station_id': station_ids,
    'predicted_demand': predicted_demand,
    'base_capacity': base_capacity
})

# 保存到Excel文件，模拟题目中可能提供的输入数据
# 假设这个文件是 'station_data_input.xlsx'
station_data.to_excel('station_data_input.xlsx', index=False)

print(f"已生成包含 {num_stations} 个站点的模拟数据，并保存至 'station_data_input.xlsx'。")
print(station_data.head())

# --- 2. 定义优化问题的参数 ---
# 这些参数将在优化模型中使用

# 最小和最大容量限制
C_min = 10
C_max = 200

# 目标函数权重
W1 = 1.0  # 借还失败惩罚权重
W2 = 1.0  # 运营成本权重

# 调度成本系数 (K_调度 in the model)
K_scheduling = 1.0

print(f"\n优化问题参数设定:")
print(f"  站点容量范围: [{C_min}, {C_max}]")
print(f"  借还失败惩罚权重 (W1): {W1}")
print(f"  运营成本权重 (W2): {W2}")
print(f"  调度成本系数 (K_scheduling): {K_scheduling}")