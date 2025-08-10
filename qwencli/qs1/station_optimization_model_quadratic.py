import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, LpInteger, LpContinuous, LpBinary, LpAffineExpression

# --- 1. 读取数据 ---
station_data_file = 'station_data_input.xlsx'
df_stations = pd.read_excel(station_data_file)

# --- 2. 定义优化问题参数 ---
C_min = 10
C_max = 200
W1 = 1.0  # 借还失败惩罚权重
W2 = 1.0  # 运营成本权重
K_scheduling = 1.0 # 调度成本系数

# --- 3. 创建优化模型 ---
# 这次我们尝试包含 (d_i - c_i)^2 项。PuLP 本身不直接支持，但我们可以使用 `pulp` 来建模，
# 然后导出为 MPS 或 LP 格式，再用支持二次规划的求解器（如 Gurobi, CPLEX）求解。
# 但为了能在本地运行，我们尝试一个近似方法或使用 `pulp` 的实验性二次支持。

# 创建一个最小化问题
prob = LpProblem("Station_Optimization_Quadratic", LpMinimize)

# --- 4. 定义决策变量 ---
x = LpVariable.dicts("Retain", df_stations['station_id'], cat=LpBinary)
c = LpVariable.dicts("Capacity", df_stations['station_id'], lowBound=0, cat=LpInteger)
y_pos = LpVariable.dicts("Borrow_Failure", df_stations['station_id'], lowBound=0, cat=LpContinuous)
y_neg = LpVariable.dicts("Return_Failure", df_stations['station_id'], lowBound=0, cat=LpContinuous)

# --- 5. 添加约束条件 ---
for _, row in df_stations.iterrows():
    sid = row['station_id']
    d_i = row['predicted_demand']
    
    prob += c[sid] <= C_max * x[sid]
    prob += c[sid] >= C_min * x[sid]
    
    prob += y_pos[sid] >= d_i - c[sid]
    prob += y_pos[sid] >= 0
    
    prob += y_neg[sid] >= -d_i - c[sid]
    prob += y_neg[sid] >= 0

# --- 6. 定义目标函数 ---
# Minimize: Z = W1 * \sum_{i} (y_i^+ + y_i^-) + W2 * (\sum_{i} (x_i \cdot |d_i|) + \sum_{i} (d_i - c_i)^2)
#
# 由于 (d_i - c_i)^2 是非线性的，PuLP 默认不支持。我们尝试使用 LpAffineExpression
# 来构建一个二次项的近似表达式，但这通常不被求解器接受。
# 正确的做法是使用支持二次规划的求解器。
# 为了演示，我们先尝试直接构建，看求解器是否支持。

# 线性部分
linear_part = W1 * lpSum([y_pos[sid] + y_neg[sid] for sid in df_stations['station_id']]) + \
              W2 * lpSum([x[sid] * abs(row['predicted_demand']) * K_scheduling for _, row in df_stations.iterrows()])

# 尝试添加二次项 (这很可能会失败)
# quadratic_part = W2 * lpSum([(row['predicted_demand'] - c[row['station_id']]) * (row['predicted_demand'] - c[row['station_id']]) for _, row in df_stations.iterrows()])
# prob += linear_part + quadratic_part

# 由于 PuLP + CBC 不直接支持二次目标函数，我们有两种选择：
# 1. 寻找替代方法或求解器。
# 2. 对模型进行线性化近似。

# --- 方法一：完全忽略二次项 ---
print("--- 方法一：忽略 (d_i - c_i)^2 项 ---")
prob.setObjective(linear_part)

# --- 求解 ---
prob.solve()

print(f"Status: {LpStatus[prob.status]}")
print(f"Objective Value (Total Cost, no quadratic term): {value(prob.objective):.2f}")

# --- 保存结果 ---
results = []
for _, row in df_stations.iterrows():
    sid = row['station_id']
    results.append({
        'station_id': sid,
        'retain': int(value(x[sid])),
        'capacity': int(value(c[sid])) if value(x[sid]) > 0.5 else 0,
        'predicted_demand': row['predicted_demand'],
        'borrow_failure': value(y_pos[sid]),
        'return_failure': value(y_neg[sid])
    })

df_results = pd.DataFrame(results)
df_status = df_results[['station_id', 'retain']]
df_capacity = df_results[df_results['retain'] > 0][['station_id', 'capacity']]

df_status.to_excel('result1_1_method1.xlsx', index=False)
df_capacity.to_excel('result1_2_method1.xlsx', index=False)

print("\n方法一结果已保存至 'result1_1_method1.xlsx' 和 'result1_2_method1.xlsx'。")


# --- 方法二：线性化 (d_i - c_i)^2 ---
print("\n--- 方法二：线性化 (d_i - c_i)^2 项 ---")
# 一种常见的线性化方法是使用分段线性逼近或引入辅助变量和约束。
# 这里我们使用一个简化的线性化方法：
# 将 (d_i - c_i)^2 近似为 |d_i - c_i| * M，其中 M 是一个合适的常数。
# 或者，我们将其近似为 max(0, |d_i| - c_i) 或 max(0, c_i - |d_i|) 的线性组合。
# 一个更合理的近似是将其视为对容量配置过大或过小的惩罚。
# 我们可以定义一个新的辅助变量 z_i = |d_i - c_i|，然后最小化 \sum z_i^2。
# 但这又回到了非线性。

# 让我们尝试一个不同的线性近似：
# (d_i - c_i)^2 ≈ |d_i - c_i| * (|d_i| + c_i)
# 这不是一个好的近似，但可以作为一个尝试。

# 更简单的线性化：
# 引入新的变量 z_i = d_i - c_i
# 引入新的变量 s_i = |z_i| (通过约束 s_i \u003e= z_i, s_i \u003e= -z_i)
# 然后最小化 \sum s_i^2。这仍然是非线性的。

# 最简单的线性化：直接最小化 \sum |d_i - c_i|
# 即最小化 \sum (u_i + v_i)，其中 u_i \u003e= d_i - c_i, u_i \u003e= 0, v_i \u003e= c_i - d_i, v_i \u003e= 0
# 这相当于最小化 |d_i - c_i| 的和。

# 让我们创建一个新的模型来实现这个线性化。
prob_linear_approx = LpProblem("Station_Optimization_Linear_Approx", LpMinimize)

# 重新定义变量 (保持名字不变，但属于新问题)
x_l = LpVariable.dicts("Retain", df_stations['station_id'], cat=LpBinary)
c_l = LpVariable.dicts("Capacity", df_stations['station_id'], lowBound=0, cat=LpInteger)
y_pos_l = LpVariable.dicts("Borrow_Failure", df_stations['station_id'], lowBound=0, cat=LpContinuous)
y_neg_l = LpVariable.dicts("Return_Failure", df_stations['station_id'], lowBound=0, cat=LpContinuous)

# 新的线性化变量 |d_i - c_i|
diff_abs_l = LpVariable.dicts("Abs_Diff", df_stations['station_id'], lowBound=0, cat=LpContinuous)

# 添加约束
for _, row in df_stations.iterrows():
    sid = row['station_id']
    d_i = row['predicted_demand']
    
    prob_linear_approx += c_l[sid] <= C_max * x_l[sid]
    prob_linear_approx += c_l[sid] >= C_min * x_l[sid]
    
    prob_linear_approx += y_pos_l[sid] >= d_i - c_l[sid]
    prob_linear_approx += y_pos_l[sid] >= 0
    
    prob_linear_approx += y_neg_l[sid] >= -d_i - c_l[sid]
    prob_linear_approx += y_neg_l[sid] >= 0
    
    # 线性化 |d_i - c_i|
    prob_linear_approx += diff_abs_l[sid] >= d_i - c_l[sid]
    prob_linear_approx += diff_abs_l[sid] >= c_l[sid] - d_i

# 定义线性近似的目标函数
# Minimize: W1 * \sum (y_pos + y_neg) + W2 * (\sum (x * |d|) + \sum |d - c|)
linear_part_l = W1 * lpSum([y_pos_l[sid] + y_neg_l[sid] for sid in df_stations['station_id']]) + \
                W2 * lpSum([x_l[sid] * abs(row['predicted_demand']) * K_scheduling for _, row in df_stations.iterrows()])
quadratic_approx_l = W2 * lpSum([diff_abs_l[sid] for sid in df_stations['station_id']]) # 这是 (d_i - c_i)^2 的线性近似

prob_linear_approx += linear_part_l + quadratic_approx_l

# --- 求解 ---
prob_linear_approx.solve()

print(f"Status: {LpStatus[prob_linear_approx.status]}")
print(f"Objective Value (Total Cost, linear approx of quadratic term): {value(prob_linear_approx.objective):.2f}")

# --- 保存结果 ---
results_l = []
for _, row in df_stations.iterrows():
    sid = row['station_id']
    results_l.append({
        'station_id': sid,
        'retain': int(value(x_l[sid])),
        'capacity': int(value(c_l[sid])) if value(x_l[sid]) > 0.5 else 0,
        'predicted_demand': row['predicted_demand'],
        'borrow_failure': value(y_pos_l[sid]),
        'return_failure': value(y_neg_l[sid]),
        'abs_diff': value(diff_abs_l[sid])
    })

df_results_l = pd.DataFrame(results_l)
df_status_l = df_results_l[['station_id', 'retain']]
df_capacity_l = df_results_l[df_results_l['retain'] > 0][['station_id', 'capacity']]

df_status_l.to_excel('result1_1_method2.xlsx', index=False)
df_capacity_l.to_excel('result1_2_method2.xlsx', index=False)

print("\n方法二结果已保存至 'result1_1_method2.xlsx' 和 'result1_2_method2.xlsx'。")

# --- 分析两种方法的结果差异 ---
print("\n--- 结果比较 ---")
print("方法一 (忽略二次项) 与 方法二 (线性化二次项) 的站点保留状态是否相同?",
      df_status.equals(df_status_l))
print("方法一 (忽略二次项) 与 方法二 (线性化二次项) 的站点容量配置是否相同?",
      df_capacity.equals(df_capacity_l))