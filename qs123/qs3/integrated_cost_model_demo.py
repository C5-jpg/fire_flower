import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, LpInteger, LpContinuous, LpBinary, LpAffineExpression

# --- 1. 读取基础数据 ---
station_data_file = 'station_data_input.xlsx'
df_stations = pd.read_excel(station_data_file)

# --- 2. 定义优化问题参数 ---
C_min = 10
C_max = 200

# 从问题一和问题二继承的参数
K_scheduling_base = 1.0 # 基础调度成本系数
P_user_penalty = 15.0   # 用户体验损失的单位惩罚成本 (根据综合成本效益模型文档设定)

# --- 3. 创建优化模型 ---
# 目标: min Z_social = C_dispatch + C_vacancy + C_user_exp
# 其中 C_user_exp 包含非光滑 max(0, |D| - c) 项，我们将其线性化

prob_integrated = LpProblem("Integrated_Cost_Benefit_Optimization", LpMinimize)

# --- 4. 定义决策变量 ---
stations = df_stations['station_id'].tolist()
# 4.1. 站点保留状态 (0-1变量)
x = LpVariable.dicts("Retain", stations, cat=LpBinary)

# 4.2. 站点容量配置 (整数变量)
c = LpVariable.dicts("Capacity", stations, lowBound=0, cat=LpInteger)

# 4.3. 辅助变量：用户体验损失 (连续变量) - 线性化 max(0, |D_i| - c_i)
u = LpVariable.dicts("User_Experience_Loss", stations, lowBound=0, cat=LpContinuous)

# --- 5. 添加约束条件 ---
# 5.1. 容量与站点状态的关系
for sid in stations:
    prob_integrated += c[sid] <= C_max * x[sid]
    prob_integrated += c[sid] >= C_min * x[sid]

# 5.2. 线性化用户体验损失 u_i = max(0, |D_i| - c_i)
for _, row in df_stations.iterrows():
    sid = row['station_id']
    D_i = row['predicted_demand']
    abs_D_i = abs(D_i)
    # u_i >= |D_i| - c_i
    prob_integrated += u[sid] >= abs_D_i - c[sid]
    # u_i >= 0 (已由变量定义保证)

# --- 6. 定义目标函数 ---
# Z_social = C_dispatch + C_vacancy + C_user_exp
# C_dispatch = sum(K_scheduling_base * |D_i| * x_i)
# C_vacancy = sum((D_i - c_i)^2) # 这是非线性的，PuLP无法直接处理
# C_user_exp = P_user_penalty * sum(u_i)
#
# 由于 (D_i - c_i)^2 是非线性的，我们有两种处理方式：
# 1.  使用支持二次规划的求解器 (如 Gurobi, CPLEX)。
# 2.  将其近似或忽略。
#
# 为了能在本地用 PuLP 演示，我们先尝试忽略二次项。
# 在后续可以探索使用其他库来处理。

# 方法一：忽略 C_vacancy 项
print("--- 方法一：忽略空置损失 (C_vacancy) 项 ---")
# C_dispatch = sum(K_scheduling_base * |D_i| * x_i)
C_dispatch_expr = lpSum([K_scheduling_base * abs(row['predicted_demand']) * x[row['station_id']] for _, row in df_stations.iterrows()])

# C_user_exp = P_user_penalty * sum(u_i)
C_user_exp_expr = P_user_penalty * lpSum([u[sid] for sid in stations])

# 注意：这里我们忽略了 C_vacancy = sum((D_i - c_i)^2)
prob_integrated.setObjective(C_dispatch_expr + C_user_exp_expr)

# --- 7. 求解 ---
prob_integrated.solve()

print(f"Status: {LpStatus[prob_integrated.status]}")
print(f"Objective Value (忽略 C_vacancy): {value(prob_integrated.objective):.2f}")

# --- 8. 整理并保存结果 ---
results = []
for _, row in df_stations.iterrows():
    sid = row['station_id']
    D_i = row['predicted_demand']
    results.append({
        'station_id': sid,
        'retain': int(value(x[sid])),
        'capacity': int(value(c[sid])) if value(x[sid]) > 0.5 else 0,
        'predicted_demand': D_i,
        'user_loss': value(u[sid]),
        'abs_diff_dc': abs(D_i - value(c[sid])) if value(x[sid]) > 0.5 else 0, # |D_i - c_i|
        'vacancy_cost': (D_i - value(c[sid]))**2 if value(x[sid]) > 0.5 else 0 # (D_i - c_i)^2
    })

df_results = pd.DataFrame(results)

# 计算各部分成本
total_dispatch_cost = sum(K_scheduling_base * abs(row['predicted_demand']) * row['retain'] for _, row in df_results.iterrows())
total_user_exp_cost = P_user_penalty * df_results['user_loss'].sum()
total_vacancy_cost = df_results['vacancy_cost'].sum()

print(f"\n--- 成本分解 (忽略 C_vacancy 项) ---")
print(f"  C_dispatch (调度成本): {total_dispatch_cost:.2f}")
print(f"  C_user_exp (用户体验损失): {total_user_exp_cost:.2f}")
print(f"  C_vacancy (空置损失, 未计入目标): {total_vacancy_cost:.2f}")
print(f"  Z_social (社会总成本, 忽略 C_vacancy): {total_dispatch_cost + total_user_exp_cost:.2f}")

# 保存到Excel文件
df_status = df_results[['station_id', 'retain']]
df_capacity = df_results[df_results['retain'] > 0][['station_id', 'capacity']]

df_status.to_excel('result3_1_method1.xlsx', index=False)
df_capacity.to_excel('result3_2_method1.xlsx', index=False)

print("\n方法一结果已保存至 'result3_1_method1.xlsx' 和 'result3_2_method1.xlsx'。")


# --- 方法二：使用 scipy.optimize.minimize_scalar 或其他库求解包含二次项的完整模型 ---
# 由于 PuLP 的限制，我们尝试一个更精确的方法来处理 (D_i - c_i)^2 项。
# 这需要将问题重新表述为一个更复杂的优化问题，或者使用专门的优化库。
# 为简化，我们尝试手动计算 (D_i - c_i)^2 项并将其加入目标函数，但这在 PuLP 中是不允许的。
# 我们可以尝试用一个近似的线性项来代替它，例如 abs(D_i - c_i) * M。
# 但这不是一个好的近似。
# 最直接的方法是承认 PuLP 的限制，并说明为了求解完整模型，需要使用更高级的工具。

print("\n--- 方法二：探讨完整模型的求解 ---")
print("完整模型包含 (D_i - c_i)^2 项，这是一个非线性项。")
print("PuLP 无法直接处理此类目标函数。")
print("为了求解完整模型，建议使用支持二次规划的求解器，例如：")
print("  - Gurobi (商业，性能卓越)")
print("  - CPLEX (商业，性能卓越)")
print("  - 使用 Python 的 `cvxpy` 或 `scipy.optimize` 等库。")
print("\n在本演示中，我们仅展示了忽略该非线性项的求解过程。")
print("在实际应用中，应使用专业工具求解完整模型以获得最优解。")

# --- 9. 与问题一结果进行比较 ---
# 比较方法一的结果与问题一方法二的结果
try:
    df_status_p1 = pd.read_excel('result1_1_method2.xlsx')
    df_capacity_p1 = pd.read_excel('result1_2_method2.xlsx')
    
    print("\n--- 与问题一结果 (方法二) 的比较 ---")
    
    # 比较保留状态
    df_comparison_status = pd.merge(df_status, df_status_p1, on='station_id', how='outer', suffixes=('_p3', '_p1'))
    df_comparison_status['same'] = df_comparison_status['retain_p3'] == df_comparison_status['retain_p1']
    num_same_status = df_comparison_status['same'].sum()
    print(f"保留状态相同的站点数: {num_same_status} / {len(stations)}")
    
    # 比较容量配置 (只比较都被保留的站点)
    retained_p3 = set(df_status[df_status['retain']==1]['station_id'])
    retained_p1 = set(df_capacity_p1['station_id'])
    common_retained = retained_p3.intersection(retained_p1)
    
    if common_retained:
        df_cap_p3 = df_capacity.set_index('station_id')
        df_cap_p1 = df_capacity_p1.set_index('station_id')
        df_common = pd.DataFrame(index=list(common_retained))
        df_common['capacity_p3'] = df_cap_p3['capacity']
        df_common['capacity_p1'] = df_cap_p1['capacity']
        df_common['diff'] = abs(df_common['capacity_p3'] - df_common['capacity_p1'])
        
        mean_diff = df_common['diff'].mean()
        print(f"共同保留站点的平均容量差异: {mean_diff:.2f}")
        print("\n前10个共同保留站点的容量比较:")
        print(df_common.head(10))
    else:
        print("没有共同保留的站点。")
        
except FileNotFoundError:
    print("\n未找到问题一的结果文件 'result1_1_method2.xlsx' 或 'result1_2_method2.xlsx' 进行比较。")