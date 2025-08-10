import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, LpInteger, LpContinuous, LpBinary

# --- 1. 读取数据 ---
station_data_file = 'station_data_input.xlsx'
df_stations = pd.read_excel(station_data_file)

# --- 2. 定义优化问题参数 ---
# 从数据生成脚本中获取
C_min = 10
C_max = 200
W1 = 1.0  # 借还失败惩罚权重
W2 = 1.0  # 运营成本权重
K_scheduling = 1.0 # 调度成本系数

# --- 3. 创建优化模型 ---
# 由于目标函数包含 (d_i - c_i)^2 项，这是一个二次规划问题。
# PuLP 本身不直接支持二次目标函数。我们可以使用 `pulp` 来建模，然后使用支持二次规划的求解器。
# 但是，为了简化，我们先移除二次项，构建一个线性规划模型。
# 我们将 (d_i - c_i)^2 近似为 |d_i - c_i| * M 或者忽略它，或者将其线性化。
# 这里我们选择忽略 (d_i - c_i)^2 项，专注于主要的线性部分。

# 创建一个最小化问题
prob = LpProblem("Station_Optimization", LpMinimize)

# --- 4. 定义决策变量 ---
# 4.1. 站点保留状态 (0-1变量)
x = LpVariable.dicts("Retain", df_stations['station_id'], cat=LpBinary)

# 4.2. 站点容量配置 (整数变量)
c = LpVariable.dicts("Capacity", df_stations['station_id'], lowBound=0, cat=LpInteger)

# 4.3. 辅助变量：借车失败惩罚 (连续变量)
y_pos = LpVariable.dicts("Borrow_Failure", df_stations['station_id'], lowBound=0, cat=LpContinuous)

# 4.4. 辅助变量：还车失败惩罚 (连续变量)
y_neg = LpVariable.dicts("Return_Failure", df_stations['station_id'], lowBound=0, cat=LpContinuous)

# --- 5. 添加约束条件 ---
for _, row in df_stations.iterrows():
    sid = row['station_id']
    d_i = row['predicted_demand']
    
    # 5.1. 容量与站点状态的关系
    prob += c[sid] <= C_max * x[sid]
    prob += c[sid] >= C_min * x[sid]
    
    # 5.2. 线性化 max(0, d_i - c_i) 和 max(0, -d_i - c_i)
    prob += y_pos[sid] >= d_i - c[sid]
    prob += y_pos[sid] >= 0
    
    prob += y_neg[sid] >= -d_i - c[sid]
    prob += y_neg[sid] >= 0

# --- 6. 定义目标函数 (移除了 (d_i - c_i)^2 项) ---
# Minimize: Z = W1 * \\sum_{i} (y_i^+ + y_i^-) + W2 * \\sum_{i} (x_i \\cdot |d_i|)
# 注意：这里的 |d_i| 对于每个站点是常数，所以 x_i \\cdot |d_i| 是一个线性项。
prob += W1 * lpSum([y_pos[sid] + y_neg[sid] for sid in df_stations['station_id']]) + \
        W2 * lpSum([x[sid] * abs(row['predicted_demand']) * K_scheduling for _, row in df_stations.iterrows()])

# --- 7. 求解 ---
# 使用默认求解器 (通常是CBC)
prob.solve()

# --- 8. 输出结果 ---
print(f"Status: {LpStatus[prob.status]}")
print(f"Objective Value (Total Cost): {value(prob.objective):.2f}")

# --- 9. 整理并保存结果 ---
results = []
for _, row in df_stations.iterrows():
    sid = row['station_id']
    results.append({
        'station_id': sid,
        'retain': int(value(x[sid])),
        'capacity': int(value(c[sid])) if value(x[sid]) > 0.5 else 0, # 如果关闭，容量为0
        'predicted_demand': row['predicted_demand'],
        'borrow_failure': value(y_pos[sid]),
        'return_failure': value(y_neg[sid])
    })

df_results = pd.DataFrame(results)

# 保存到Excel文件
# result1_1.xlsx: 站点的保留/关闭状态
df_status = df_results[['station_id', 'retain']]
df_status.to_excel('result1_1.xlsx', index=False)

# result1_2.xlsx: 保留站点的容量配置
# 只输出被保留的站点
df_capacity = df_results[df_results['retain'] > 0][['station_id', 'capacity']]
df_capacity.to_excel('result1_2.xlsx', index=False)

print("\n优化结果已保存至 'result1_1.xlsx' 和 'result1_2.xlsx'。")
print(df_results.head(10))