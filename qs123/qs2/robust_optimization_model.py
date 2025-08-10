import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, LpInteger, LpContinuous, LpBinary, LpConstraintLE, LpConstraintGE

# --- 1. 读取场景数据 ---
scenario_file = 'uncertainty_scenarios.xlsx'
df_scenarios = pd.read_excel(scenario_file)

# --- 2. 定义优化问题参数 (与问题一和场景生成保持一致) ---
C_min = 10
C_max = 200
W1 = 1.0  # 借还失败惩罚权重
W2 = 1.0  # 运营成本权重

# --- 3. 获取所有唯一站点和场景 ---
stations = df_scenarios['station_id'].unique().tolist()
scenarios = df_scenarios['scenario'].unique().tolist()

# 创建一个字典来存储每个场景下每个站点的数据，方便快速访问
scenario_data_dict = {}
for _, row in df_scenarios.iterrows():
    sid = row['station_id']
    sc = row['scenario']
    if sc not in scenario_data_dict:
        scenario_data_dict[sc] = {}
    scenario_data_dict[sc][sid] = {
        'actual_demand': row['actual_demand'],
        'actual_cost_coeff': row['actual_cost_coeff']
    }

# --- 4. 创建鲁棒优化模型 ---
# 目标: min_{x,c} max_{s in S} Cost(x, c, s)
# 通过引入一个辅助变量 eta 来表示最坏情况下的成本，将问题转化为:
# min eta
# s.t. Cost(x, c, s) <= eta  for all s in S
#       (x, c) 满足问题一中的约束

# 创建最小化问题
prob_robust = LpProblem("Robust_Station_Optimization", LpMinimize)

# --- 5. 定义决策变量 ---
# 5.1. 站点保留状态 (0-1变量) - 这是我们的主要决策变量
x = LpVariable.dicts("Retain", stations, cat=LpBinary)

# 5.2. 站点容量配置 (整数变量) - 这是我们的主要决策变量
c = LpVariable.dicts("Capacity", stations, lowBound=0, cat=LpInteger)

# 5.3. 辅助变量：最坏情况下的总成本
eta = LpVariable("Worst_Case_Cost", lowBound=0, cat=LpContinuous)

# 5.4. 辅助变量：每个场景下的借车失败惩罚 (连续变量)
y_pos = LpVariable.dicts("Borrow_Failure", (scenarios, stations), lowBound=0, cat=LpContinuous)

# 5.5. 辅助变量：每个场景下的还车失败惩罚 (连续变量)
y_neg = LpVariable.dicts("Return_Failure", (scenarios, stations), lowBound=0, cat=LpContinuous)

# --- 6. 添加约束条件 ---

# 6.1. 对于每个站点，添加容量与保留状态的约束 (这些约束不依赖于场景)
for sid in stations:
    prob_robust += c[sid] <= C_max * x[sid]
    prob_robust += c[sid] >= C_min * x[sid]

# 6.2. 对于每个场景，添加成本约束和辅助变量的约束
for sc in scenarios:
    # 6.2.1. 对于每个站点在该场景下，线性化 max(0, d~_i - c_i) 和 max(0, -d~_i - c_i)
    for sid in stations:
        d_tilde = scenario_data_dict[sc][sid]['actual_demand']
        prob_robust += y_pos[sc][sid] >= d_tilde - c[sid]
        prob_robust += y_pos[sc][sid] >= 0
        
        prob_robust += y_neg[sc][sid] >= -d_tilde - c[sid]
        prob_robust += y_neg[sc][sid] >= 0
    
    # 6.2.2. 计算该场景下的总成本表达式
    # Cost_sc = W1 * \sum_{i} (y_pos[sc][i] + y_neg[sc][i]) + W2 * \sum_{i} (x[i] * K~_scheduling,i * |d~_i|)
    cost_expr_sc = W1 * lpSum([y_pos[sc][sid] + y_neg[sc][sid] for sid in stations]) + \
                   W2 * lpSum([x[sid] * scenario_data_dict[sc][sid]['actual_cost_coeff'] * abs(scenario_data_dict[sc][sid]['actual_demand']) for sid in stations])
    
    # 6.2.3. 添加约束：该场景下的成本 <= 最坏情况成本 eta
    prob_robust += cost_expr_sc <= eta

# --- 7. 定义目标函数 ---
# 最小化最坏情况下的成本
prob_robust += eta

# --- 8. 求解 ---
# 使用默认求解器 (通常是CBC)
prob_robust.solve()

# --- 9. 输出结果 ---
print(f"Status: {LpStatus[prob_robust.status]}")
print(f"Robust Objective Value (Worst-Case Cost): {value(prob_robust.objective):.2f}")

# --- 10. 分析哪个场景是当前解的最坏情况 ---
worst_scenario = None
max_cost = -float('inf')
scenario_costs = {}

# 在计算场景成本之前，先打印一些决策变量的值用于调试
print("\n--- 调试信息 ---")
print(f"前10个站点的保留状态 x[sid]: {[value(x[sid]) for sid in stations[:10]]}")
print(f"前10个站点的容量配置 c[sid]: {[value(c[sid]) for sid in stations[:10]]}")

for sc in scenarios:
    # 重新显式计算成本，确保与约束中的表达式一致
    cost_sc = 0.0
    # 第一部分：失败惩罚
    part1 = 0.0
    for sid in stations:
        part1 += value(y_pos[sc][sid]) + value(y_neg[sc][sid])
    part1 *= W1
    
    # 第二部分：运营成本
    part2 = 0.0
    for sid in stations:
        part2 += value(x[sid]) * scenario_data_dict[sc][sid]['actual_cost_coeff'] * abs(scenario_data_dict[sc][sid]['actual_demand'])
    part2 *= W2
    
    cost_sc = part1 + part2
    scenario_costs[sc] = cost_sc
    if cost_sc is not None and cost_sc > max_cost:
        max_cost = cost_sc
        worst_scenario = sc

print(f"\n在最优鲁棒解下，最坏情况发生在场景: {worst_scenario} (Cost: {max_cost:.2f})")
for sc, cost in scenario_costs.items():
    print(f"  - {sc}: Cost = {cost:.2f}")

# --- 11. 整理并保存鲁棒优化结果 ---
results_robust = []
for sid in stations:
    results_robust.append({
        'station_id': sid,
        'retain': int(value(x[sid])),
        'capacity': int(value(c[sid])) if value(x[sid]) > 0.5 else 0,
    })

df_results_robust = pd.DataFrame(results_robust)

# 保存到Excel文件 (与问题一的格式保持一致)
# result2_1.xlsx: 站点的保留/关闭状态
df_status_robust = df_results_robust[['station_id', 'retain']]
df_status_robust.to_excel('result2_1.xlsx', index=False)

# result2_2.xlsx: 保留站点的容量配置
df_capacity_robust = df_results_robust[df_results_robust['retain'] > 0][['station_id', 'capacity']]
df_capacity_robust.to_excel('result2_2.xlsx', index=False)

print("\n鲁棒优化结果已保存至 'result2_1.xlsx' 和 'result2_2.xlsx'。")

# --- 12. 与问题一的结果进行比较 ---
# 读取问题一方法二（线性化二次项）的结果作为对比
try:
    df_status_det = pd.read_excel('../result1_1_method2.xlsx')
    df_capacity_det = pd.read_excel('../result1_2_method2.xlsx')
    
    print("\n--- 与问题一确定性解 (方法二) 的比较 ---")
    
    # 比较保留状态
    df_comparison_status = pd.merge(df_status_robust, df_status_det, on='station_id', how='outer', suffixes=('_robust', '_deterministic'))
    df_comparison_status['same'] = df_comparison_status['retain_robust'] == df_comparison_status['retain_deterministic']
    num_same_status = df_comparison_status['same'].sum()
    print(f"保留状态相同的站点数: {num_same_status} / {len(stations)}")
    
    # 比较容量配置 (只比较在两种方案中都被保留的站点)
    # 首先确定在鲁棒解和确定性解中分别保留了哪些站点
    retained_robust = set(df_status_robust[df_status_robust['retain'] == 1]['station_id'])
    retained_det = set(df_capacity_det['station_id'])
    
    # 找到在两种方案中都被保留的站点
    common_retained = retained_robust.intersection(retained_det)
    
    if not df_capacity_robust.empty and not df_capacity_det.empty:
        df_comparison_capacity = pd.merge(df_capacity_robust, df_capacity_det, on='station_id', how='outer', suffixes=('_robust', '_deterministic'))
        # 只计算在两种方案中都被保留的站点的容量差异
        df_comparison_capacity_common = df_comparison_capacity[df_comparison_capacity['station_id'].isin(common_retained)]
        if not df_comparison_capacity_common.empty:
            df_comparison_capacity_common['diff'] = (df_comparison_capacity_common['capacity_robust'] - df_comparison_capacity_common['capacity_deterministic']).abs()
            mean_capacity_diff = df_comparison_capacity_common['diff'].mean()
            print(f"在两种方案中都被保留的站点 ({len(common_retained)} 个) 的平均容量配置差异: {mean_capacity_diff:.2f}")
            print("\n这些共同保留站点的容量比较 (前10个):")
            print(df_comparison_capacity_common.head(10)[['station_id', 'capacity_robust', 'capacity_deterministic', 'diff']])
        else:
            print(f"没有站点在两种方案中都被保留，无法比较容量配置。")
    else:
        print(f"鲁棒解或确定性解没有保留任何站点，无法比较容量配置。")
        print(f"  - 鲁棒解保留的站点数: {len(retained_robust)}")
        print(f"  - 确定性解保留的站点数: {len(retained_det)}")
    
except FileNotFoundError:
    print("\n未找到问题一的结果文件 '../result1_1_method2.xlsx' 或 '../result1_2_method2.xlsx' 进行比较。")