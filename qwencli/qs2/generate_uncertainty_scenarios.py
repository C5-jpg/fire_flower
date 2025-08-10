import pandas as pd
import numpy as np

# --- 1. 读取问题一生成的基础数据 ---
input_file = '../station_data_input.xlsx'
df_stations = pd.read_excel(input_file)

# --- 2. 定义不确定性参数 ---
# 需求波动率 (±15%)
demand_fluctuation = 0.15

# 地铁影响因子区间 [-15%, 25%]
metro_impact_lower = -0.15
metro_impact_upper = 0.25

# 成本上涨率 (+5%)
cost_inflation = 0.05

# --- 3. 生成不确定性场景 ---
# 为了简化，我们采用场景近似法 (Scenario Approximation)
# 生成几个代表性的"最坏情况"场景

# 场景1: 需求向上波动最大 + 地铁负面影响最大 + 成本上涨
# 场景2: 需求向下波动最大 + 地铁正面影响最大
# 场景3: 需求向上波动中等 + 地铁负面影响中等
# 场景4: 需求向下波动中等 + 地铁正面影响中等
# 场景5: 基准场景 (预测值)

scenarios = {
    'Scenario_1_Worst': {
        'demand_factor': 1 + demand_fluctuation, # 需求增加15%
        'metro_factor': 1 + metro_impact_lower,  # 地铁负面影响15%
        'cost_factor': 1 + cost_inflation        # 成本增加5%
    },
    'Scenario_2_Best': {
        'demand_factor': 1 - demand_fluctuation, # 需求减少15%
        'metro_factor': 1 + metro_impact_upper,  # 地铁正面影响25%
        'cost_factor': 1                         # 成本无变化
    },
    'Scenario_3_Mixed_A': {
        'demand_factor': 1 + demand_fluctuation * 0.7, # 需求增加10.5%
        'metro_factor': 1 + metro_impact_lower * 0.7,  # 地铁负面影响10.5%
        'cost_factor': 1 + cost_inflation * 0.7        # 成本增加3.5%
    },
    'Scenario_4_Mixed_B': {
        'demand_factor': 1 - demand_fluctuation * 0.7, # 需求减少10.5%
        'metro_factor': 1 + metro_impact_upper * 0.7,  # 地铁正面影响17.5%
        'cost_factor': 1                               # 成本无变化
    },
    'Scenario_5_Base': {
        'demand_factor': 1,  # 预测需求
        'metro_factor': 1,   # 无地铁影响
        'cost_factor': 1     # 基准成本
    }
}

# --- 4. 为每个场景计算实际需求和成本系数 ---
scenario_data_list = []

for scenario_name, factors in scenarios.items():
    df_scenario = df_stations.copy()
    df_scenario['scenario'] = scenario_name
    df_scenario['actual_demand'] = df_scenario['predicted_demand'] * factors['demand_factor'] * factors['metro_factor']
    df_scenario['actual_cost_coeff'] = 1.0 * factors['cost_factor'] # 假设基础调度成本系数为1.0
    scenario_data_list.append(df_scenario)

# 合并所有场景数据
df_all_scenarios = pd.concat(scenario_data_list, ignore_index=True)

# --- 5. 保存场景数据 ---
scenario_output_file = 'uncertainty_scenarios.xlsx'
df_all_scenarios.to_excel(scenario_output_file, index=False)

print(f"已生成 {len(scenarios)} 个不确定性场景，并保存至 '{scenario_output_file}'。")

# --- 6. 显示一个场景的示例 ---
print("\n示例：场景 'Scenario_1_Worst' 的前5行数据：")
print(df_all_scenarios[df_all_scenarios['scenario'] == 'Scenario_1_Worst'][['station_id', 'predicted_demand', 'actual_demand', 'actual_cost_coeff']].head())

# --- 7. 更新优化问题参数 ---
# 这些参数将用于后续的鲁棒优化模型
C_min = 10
C_max = 200
W1 = 1.0  # 借还失败惩罚权重
W2 = 1.0  # 运营成本权重
# K_scheduling 基础值现在是 1.0，但在每个场景中会乘以 actual_cost_coeff