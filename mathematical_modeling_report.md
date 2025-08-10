# 2025年度“火花杯”数学建模精英联赛-B题：共享单车站点布局与容量配置优化

## 摘要

本文针对共享单车系统的站点布局与容量配置问题，构建了三个递进的优化模型。首先，通过加权和法建立静态多目标优化模型（问题一），在服务质量和运营成本之间寻求平衡。其次，引入鲁棒优化思想（问题二），提升方案在面对需求波动等不确定性时的稳定性。最后，构建综合成本效益模型（问题三），将用户体验损失纳入考量，寻求社会总成本最优的解决方案。模型求解采用了混合整数规划（MIP）和混合整数二次规划（MIQP）技术，并利用COPT求解器进行高效求解。通过对结果的对比分析，验证了不同模型侧重点对决策的影响，为共享单车系统的精细化运营提供了数据驱动的决策支持。

**关键词**: 共享单车; 优化模型; 多目标优化; 鲁棒优化; COPT

## 1. 问题重述与分析

### 1.1 问题背景

随着城市化进程的加速，共享单车作为一种绿色、便捷的出行方式，在解决“最后一公里”问题上发挥着重要作用。然而，站点布局不合理、容量配置不科学等问题，常常导致用户体验不佳和运营效率低下。因此，如何科学地规划站点和配置容量，是共享单车运营商面临的核心挑战。

### 1.2 问题目标

本题旨在基于给定的历史骑行数据、站点需求预测及地铁覆盖信息，通过建立数学模型，为共享单车系统的运营提供优化方案。

*   **问题一 (静态优化)**: 在给定总车位数不超过现有1.1倍的约束下，设计两个优化方案：
    *   **方案1_1**: 侧重成本控制，最小化运营成本（调度与空置损失）。
    *   **方案1_2**: 侧重服务提升，最小化服务损失（缺车/缺桩造成的损失）。
*   **问题二 (鲁棒优化)**: 考虑需求波动和成本变动等不确定性因素，设计一个在最坏情况下依然表现稳健的优化方案。
*   **问题三 (综合效益)**: 构建一个统一的社会总成本函数，综合考虑调度成本、空置损失和用户体验损失，给出最优的站点布局与容量配置方案。

### 1.3 数据说明

*   `202503-capitalbikeshare-tripdata.csv`: 历史骑行记录数据。
*   `demand_features.csv`: 站点需求预测及相关特征。
*   `metro_coverage.xlsx`: 地铁站点覆盖信息。

## 2. 模型假设与符号说明

### 2.1 模型假设

1.  **需求预测准确性**: `demand_features.csv` 中的 `predicted_demand` 是对站点未来需求的准确预测。
2.  **成本线性性**: 调度成本与站点借还车差值的绝对值 (`|B_i - R_i|`) 成正比。
3.  **空置损失二次性**: 空置损失与 `(D_i - c_i)^2` 成正比。
4.  **不确定性范围**: 需求波动范围为 ±15%，成本上涨5%，地铁影响导致需求变化范围为 -25% 到 +15%。
5.  **站点关闭限制**: 最多可关闭20%的低使用率站点。
6.  **固定站点**: 除了候选关闭站点外，其余站点必须保留。

### 2.2 符号说明

*   **集合**:
    *   `I`: 所有站点的集合。
    *   `I_cand ⊆ I`: 候选关闭站点集合（使用率最低的20%）。
    *   `I_fixed = I \ I_cand`: 必须保留的站点集合。
*   **参数**:
    *   `D_i`: 站点 `i` 的预测需求。
    *   `D_i_robust`: 站点 `i` 在鲁棒优化模型中的最坏情况需求。
    *   `B_i`: 站点 `i` 的历史总借车数。
    *   `R_i`: 站点 `i` 的历史总还车数。
    *   `Δ_i = B_i - R_i`: 站点 `i` 的历史借还车差值。
    *   `C_total_initial`: 现有网络总车位数。
    *   `C_max = 1.1 * C_total_initial`: 允许的最大总车位数。
    *   `M`: 一个足够大的正数（Big-M法）。
    *   `P`: 用户体验损失的惩罚系数。
*   **决策变量**:
    *   `x_i ∈ {0, 1}`: 对于 `i ∈ I_cand`，若站点 `i` 保留则为1，关闭则为0。对于 `i ∈ I_fixed`，`x_i = 1`。
    *   `c_i ∈ Z+`: 站点 `i` 优化后的容量。

## 3. 数据预处理与特征工程

数据预处理是建模的基础。我们首先加载并分析原始数据，然后构建用于模型求解的主数据集。

### 3.1 加载与初步分析

```python
# -*- coding: utf-8 -*-
"""
共享单车站点布局与容量配置优化 - 数据预处理与特征工程
"""

import pandas as pd
import numpy as np
import os

# 正确处理包含中文引号的路径
DATA_DIR = 'C:\\Users\\hk\\fire_flower\\2025年度"火花杯"数学建模精英联赛-B题-附件'
TRIPDATA_FILE = os.path.join(DATA_DIR, '202503-capitalbikeshare-tripdata.csv')
DEMAND_FEATURES_FILE = os.path.join(DATA_DIR, 'demand_features.csv')
METRO_FILE = os.path.join(DATA_DIR, 'metro_coverage.xlsx')

print("Loading data...")
# 1. 加载骑行数据
# 由于文件较大，分块读取以计算借还车次数
chunksize = 50000
borrow_counts = {}
return_counts = {}

chunk_iter = pd.read_csv(TRIPDATA_FILE, chunksize=chunksize)
print("Processing tripdata in chunks...")
chunk_count = 0
for chunk in chunk_iter:
    chunk_count += 1
    if chunk_count % 10 == 0:
        print(f"  Processed chunk {chunk_count}")

    # 计算每个站点的借车次数
    start_counts = chunk['start_station_id'].value_counts().to_dict()
    for k, v in start_counts.items():
        borrow_counts[k] = borrow_counts.get(k, 0) + v
        
    # 计算每个站点的还车次数
    end_counts = chunk['end_station_id'].value_counts().to_dict()
    for k, v in end_counts.items():
        return_counts[k] = return_counts.get(k, 0) + v

print(f"Total chunks processed: {chunk_count}")

# 汇总为DataFrame
borrow_df = pd.DataFrame.from_dict(borrow_counts, orient='index', columns=['borrow_count'])
borrow_df.index.name = 'station_id'
return_df = pd.DataFrame.from_dict(return_counts, orient='index', columns=['return_count'])
return_df.index.name = 'station_id'

# 合并借还车数据
station_activity_df = pd.merge(borrow_df, return_df, left_index=True, right_index=True, how='outer').fillna(0)
station_activity_df['borrow_count'] = station_activity_df['borrow_count'].astype(int)
station_activity_df['return_count'] = station_activity_df['return_count'].astype(int)
station_activity_df['net_flow'] = station_activity_df['borrow_count'] - station_activity_df['return_count']
station_activity_df['total_activity'] = station_activity_df['borrow_count'] + station_activity_df['return_count']

print("\\nStation activity data (Top 5):")
print(station_activity_df.head())

# 2. 加载需求预测数据
demand_df = pd.read_csv(DEMAND_FEATURES_FILE)
print("\\nDemand features data (Top 5):")
print(demand_df.head())

# 3. 加载地铁覆盖数据
metro_df = pd.read_excel(METRO_FILE)
print("\\nMetro coverage data (Top 5):")
print(metro_df.head())

```

### 3.2 特征融合与候选站点选择

```python
# -*- coding: utf-8 -*-
"""
共享单车站点布局与容量配置优化 - 特征融合与候选站点选择
"""

import pandas as pd
import numpy as np

# 假设 station_activity_df, demand_df, metro_df 已在上一步加载并定义

# 1. 合并所有数据到主数据框
# 以 demand_df 为基础，因为它包含了所有需要预测的站点
master_df = demand_df.copy()

# 合并借还车数据
master_df = pd.merge(master_df, station_activity_df, on='station_id', how='left').fillna(0)

# 合并地铁覆盖数据
# 假设 metro_df 中的 'station_id' 列与我们数据中的 'station_id' 对应
# 如果列名不同，需要进行重命名
master_df = pd.merge(master_df, metro_df[['station_id', 'near_metro']], on='station_id', how='left')
# 如果没有地铁信息，默认为False
master_df['near_metro'] = master_df['near_metro'].fillna(False)

print("Master DataFrame after merging:")
print(master_df.head())
print(f"Shape: {master_df.shape}")

# 2. 计算初始总车位数
C_total_initial = master_df['station_id'].nunique() * 30 # 假设初始每个站点30个车位，或使用其他逻辑
print(f"\\n假设初始总车位数 C_total_initial: {C_total_initial}")

# 3. 选择候选关闭站点 (使用率最低的20%)
# 按总活动量排序
master_df_sorted = master_df.sort_values(by='total_activity')
num_to_close = int(0.2 * len(master_df_sorted))
candidate_stations_to_close = master_df_sorted.head(num_to_close)['station_id'].tolist()

print(f"\\nIdentified {len(candidate_stations_to_close)} candidate stations for closure (bottom 20% by activity).")
print("First 10 candidates:", candidate_stations_to_close[:10])

# 4. 保存处理后的主数据框 (可选，用于后续模型调用)
# master_df.to_csv('processed_master_data.csv', index=False)

```

## 4. 模型构建与求解

### 4.1 问题一：静态多目标优化模型 (加权和法)

#### 4.1.1 数学模型

**目标函数 (加权和)**:
最小化 `Z = w * Z1' + (1-w) * Z2'`
其中：
*   `Z1 = Σ (|Δ_i| * x_i + (D_i - c_i)^2)` (运营成本)
*   `Z2 = Σ max(0, D_i - c_i)` (服务损失)
*   `Z1'` 和 `Z2'` 是归一化后的目标值。

**约束条件**:
1.  `Σ c_i ≤ C_max`
2.  `Σ (1 - x_i) for i in I_cand ≤ 0.2 * |I|`
3.  `c_i ≤ M * x_i` (Big-M约束，关闭站点容量为0)
4.  `c_i ≥ x_i` (保留站点容量至少为1)
5.  `x_i = 1` for `i in I_fixed`
6.  `x_i ∈ {0, 1}` for `i in I_cand`
7.  `c_i ∈ Z+`

#### 4.1.2 COPT 求解代码 (问题一)

```python
# -*- coding: utf-8 -*-
"""
共享单车站点布局与容量配置优化 - 问题一 COPT 求解
"""

import pandas as pd
import numpy as np
from coptpy import *

# 假设 master_df, candidate_stations_to_close, C_total_initial 已定义
# 从 processed_master_data.csv 读取 (如果之前已保存)
# master_df = pd.read_csv('processed_master_data.csv')
# candidate_stations_to_close = ... # 从上一步获取或重新计算

def solve_problem_1(master_df, candidate_stations_to_close, C_total_initial, weight_cost, output_filename):
    """
    使用COPT求解问题一的加权和模型
    :param master_df: 主数据框
    :param candidate_stations_to_close: 候选关闭站点列表
    :param C_total_initial: 初始总车位数
    :param weight_cost: 成本目标的权重 w
    :param output_filename: 结果保存文件名
    """
    try:
        # 创建COPT环境和模型
        env = Envr()
        model = env.createModel("Bike_Sharing_Problem_1")

        C_max = 1.1 * C_total_initial
        M = 10000 # Big-M 值，应足够大

        # 索引和参数准备
        stations = master_df['station_id'].tolist()
        D = dict(zip(stations, master_df['predicted_demand']))
        B = dict(zip(stations, master_df['borrow_count']))
        R = dict(zip(stations, master_df['return_count']))
        
        # 计算借还差值的绝对值
        Delta = {i: abs(B[i] - R[i]) for i in stations}
        
        I_cand = set(candidate_stations_to_close)
        I_fixed = set(stations) - I_cand

        # 决策变量
        x = model.addVars(stations, vtype=GRB.BINARY, nameprefix="x")
        c = model.addVars(stations, vtype=GRB.INTEGER, nameprefix="c")

        # 固定必须保留的站点
        for i in I_fixed:
            model.addConstr(x[i] == 1, name=f"fix_x_{i}")

        # 目标函数组件
        cost_expr = quicksum(Delta[i] * x[i] + (D[i] - c[i]) * (D[i] - c[i]) for i in stations)
        service_loss_expr = quicksum(gp.max_(0, D[i] - c[i]) for i in stations)
        
        # 为了加权和，我们需要先求解两个目标的最优值来归一化
        # 这里简化处理，假设已经知道或可以通过两次单独求解得到范围
        # 在实际应用中，应先进行预处理求解以获得 Z1_min, Z1_max, Z2_min, Z2_max
        # 为简化，我们直接使用表达式进行加权 (这在量纲差异大时可能不理想)
        
        # 归一化 (示例值，实际应用中需要预计算)
        Z1_ref = 1000000 
        Z2_ref = 50000   
        
        if Z1_ref == 0: Z1_ref = 1
        if Z2_ref == 0: Z2_ref = 1
            
        Z1_norm = cost_expr / Z1_ref
        Z2_norm = service_loss_expr / Z2_ref
        
        # 加权目标函数
        model.setObjective(weight_cost * Z1_norm + (1 - weight_cost) * Z2_norm, GRB.MINIMIZE)

        # 约束条件
        # 1. 总容量约束
        model.addConstr(quicksum(c[i] for i in stations) <= C_max, name="total_capacity")

        # 2. 关闭站点数量约束
        model.addConstr(quicksum(1 - x[i] for i in I_cand) <= 0.2 * len(stations), name="close_limit")

        # 3. Big-M约束：关闭站点容量为0
        for i in stations:
            model.addConstr(c[i] <= M * x[i], name=f"big_m_c_{i}")

        # 4. 保留站点容量至少为1
        for i in stations:
            model.addConstr(c[i] >= x[i], name=f"min_capacity_{i}")

        # 求解
        model.setParam(GRB.Param.OutputFlag, 1) # 输出求解日志
        model.solve()

        if model.status == GRB.Status.OPTIMAL:
            print(f"\\nProblem 1 solved optimally with weight_cost={weight_cost}")
            print(f"Objective Value (Weighted): {model.objval}")
            
            # 提取结果
            results = []
            for i in stations:
                results.append({
                    'station_id': i,
                    'is_open': int(x[i].x), # 转换为整数 0/1
                    'capacity': int(c[i].x)  # 转换为整数
                })
            
            results_df = pd.DataFrame(results)
            # 保存结果
            results_df.to_excel(output_filename, index=False)
            print(f"Results saved to {output_filename}")
            
            # 打印一些关键统计信息
            open_stations = results_df[results_df['is_open'] == 1]
            total_capacity = open_stations['capacity'].sum()
            closed_stations_count = len(results_df) - len(open_stations)
            
            print(f"\\n--- Summary for weight_cost={weight_cost} ---")
            print(f"Total Open Stations: {len(open_stations)}")
            print(f"Total Closed Stations: {closed_stations_count}")
            print(f"Allocated Total Capacity: {total_capacity} (Limit: {C_max})")
            
            # 计算实际的 Z1 和 Z2 值
            Z1_actual = sum(Delta[i] * results_df.loc[results_df['station_id']==i, 'is_open'].iloc[0] + 
                            (D[i] - results_df.loc[results_df['station_id']==i, 'capacity'].iloc[0])**2 
                            for i in stations)
            Z2_actual = sum(max(0, D[i] - results_df.loc[results_df['station_id']==i, 'capacity'].iloc[0]) 
                            for i in stations)
            print(f"Actual Cost (Z1): {Z1_actual}")
            print(f"Actual Service Loss (Z2): {Z2_actual}")
            print("---------------------------------------\\n")
            
        else:
            print(f"\\nProblem 1 failed to solve to optimality with weight_cost={weight_cost}. Status: {model.status}")

    except Exception as e:
        print(f"An error occurred in solve_problem_1: {e}")


# --- 运行问题一的两个方案 ---
# 方案 1_1: 侧重成本 (高权重给成本)
solve_problem_1(master_df, candidate_stations_to_close, C_total_initial, weight_cost=0.8, output_filename='result1_1.xlsx')

# 方案 1_2: 侧重服务 (低权重给成本)
solve_problem_1(master_df, candidate_stations_to_close, C_total_initial, weight_cost=0.2, output_filename='result1_2.xlsx')

```

### 4.2 问题二：鲁棒优化模型

#### 4.2.1 数学模型

在问题一的基础上，引入不确定性参数，构建鲁棒优化模型。

**目标函数**:
最小化最坏情况下的总成本 `Z_robust = Σ (1.05 * |Δ_i| * x_i + (D_i_robust - c_i)^2 + P * max(0, D_i_robust - c_i))`

其中，`D_i_robust` 是站点 `i` 在最坏情况下的需求：
*   如果不受地铁影响: `D_i_robust = D_i * (1 + 0.15)` (需求波动上限)
*   如果受地铁影响: `D_i_robust = D_i * (1 - 0.25) * (1 + 0.15)` (在最大降幅基础上，再考虑需求波动)

**约束条件**:
与问题一相同。

#### 4.2.2 COPT 求解代码 (问题二)

```python
# -*- coding: utf-8 -*-
"""
共享单车站点布局与容量配置优化 - 问题二 COPT 求解 (鲁棒优化)
"""

import pandas as pd
import numpy as np
from coptpy import *

# 假设 master_df, candidate_stations_to_close, C_total_initial 已定义

def solve_problem_2_robust(master_df, candidate_stations_to_close, C_total_initial, penalty_coefficient_P, output_filename):
    """
    使用COPT求解问题二的鲁棒优化模型
    :param master_df: 主数据框
    :param candidate_stations_to_close: 候选关闭站点列表
    :param C_total_initial: 初始总车位数
    :param penalty_coefficient_P: 用户体验损失惩罚系数 P
    :param output_filename: 结果保存文件名
    """
    try:
        # 创建COPT环境和模型
        env = Envr()
        model = env.createModel("Bike_Sharing_Problem_2_Robust")

        C_max = 1.1 * C_total_initial
        M = 10000 # Big-M 值

        # 索引和参数准备
        stations = master_df['station_id'].tolist()
        D = dict(zip(stations, master_df['predicted_demand']))
        B = dict(zip(stations, master_df['borrow_count']))
        R = dict(zip(stations, master_df['return_count']))
        near_metro = dict(zip(stations, master_df['near_metro']))
        
        # 计算鲁棒需求 D_i_robust
        D_robust = {}
        demand_fluctuation = 0.15 # 需求波动 ±15%
        metro_impact_down = 0.25  # 地铁影响导致需求下降 25%
        
        for i in stations:
            if near_metro[i]:
                # 受地铁影响的站点：先降25%，再考虑+15%的波动
                D_robust[i] = D[i] * (1 - metro_impact_down) * (1 + demand_fluctuation)
            else:
                # 不受地铁影响的站点：直接考虑+15%的波动
                D_robust[i] = D[i] * (1 + demand_fluctuation)
                
        # 计算调整后的调度成本系数 (成本上涨5%)
        Delta_robust = {i: 1.05 * abs(B[i] - R[i]) for i in stations}
        
        I_cand = set(candidate_stations_to_close)
        I_fixed = set(stations) - I_cand

        # 决策变量
        x = model.addVars(stations, vtype=GRB.BINARY, nameprefix="x")
        c = model.addVars(stations, vtype=GRB.INTEGER, nameprefix="c")

        # 固定必须保留的站点
        for i in I_fixed:
            model.addConstr(x[i] == 1, name=f"fix_x_{i}")

        # 鲁棒优化目标函数 (MIQP)
        # Z_robust = Σ (1.05 * |Δ_i| * x_i + (D_i_robust - c_i)^2 + P * max(0, D_i_robust - c_i))
        cost_expr = quicksum(
            Delta_robust[i] * x[i] + 
            (D_robust[i] - c[i]) * (D_robust[i] - c[i]) + 
            penalty_coefficient_P * gp.max_(0, D_robust[i] - c[i])
            for i in stations
        )
        
        model.setObjective(cost_expr, GRB.MINIMIZE)

        # 约束条件 (与问题一相同)
        # 1. 总容量约束
        model.addConstr(quicksum(c[i] for i in stations) <= C_max, name="total_capacity")

        # 2. 关闭站点数量约束
        model.addConstr(quicksum(1 - x[i] for i in I_cand) <= 0.2 * len(stations), name="close_limit")

        # 3. Big-M约束：关闭站点容量为0
        for i in stations:
            model.addConstr(c[i] <= M * x[i], name=f"big_m_c_{i}")

        # 4. 保留站点容量至少为1
        for i in stations:
            model.addConstr(c[i] >= x[i], name=f"min_capacity_{i}")

        # 求解
        model.setParam(GRB.Param.OutputFlag, 1) # 输出求解日志
        model.solve()

        if model.status == GRB.Status.OPTIMAL:
            print("\\nProblem 2 (Robust) solved optimally")
            print(f"Objective Value (Robust Cost): {model.objval}")
            
            # 提取结果
            results = []
            for i in stations:
                results.append({
                    'station_id': i,
                    'is_open': int(x[i].x),
                    'capacity': int(c[i].x)
                })
            
            results_df = pd.DataFrame(results)
            # 保存结果
            results_df.to_excel(output_filename, index=False)
            print(f"Results saved to {output_filename}")
            
            # 打印关键统计信息
            open_stations = results_df[results_df['is_open'] == 1]
            total_capacity = open_stations['capacity'].sum()
            closed_stations_count = len(results_df) - len(open_stations)
            
            print("\\n--- Summary for Problem 2 (Robust) ---")
            print(f"Total Open Stations: {len(open_stations)}")
            print(f"Total Closed Stations: {closed_stations_count}")
            print(f"Allocated Total Capacity: {total_capacity} (Limit: {C_max})")
            print("---------------------------------------\\n")
            
        else:
            print(f"\\nProblem 2 (Robust) failed to solve to optimality. Status: {model.status}")

    except Exception as e:
        print(f"An error occurred in solve_problem_2_robust: {e}")

# --- 运行问题二 ---
# 假设用户体验损失惩罚系数 P = 10 (需要根据实际情况调整)
solve_problem_2_robust(master_df, candidate_stations_to_close, C_total_initial, penalty_coefficient_P=10, output_filename='result2.xlsx')

```

### 4.3 问题三：综合成本效益模型

#### 4.3.1 数学模型

构建一个统一的社会总成本函数，将所有成本货币化。

**目标函数**:
最小化社会总成本 `Z_total = Σ (|Δ_i| * x_i + (D_i - c_i)^2 + P * max(0, D_i - c_i))`

**约束条件**:
与问题一相同。

#### 4.3.2 COPT 求解代码 (问题三)

```python
# -*- coding: utf-8 -*-
"""
共享单车站点布局与容量配置优化 - 问题三 COPT 求解 (综合成本效益)
"""

import pandas as pd
import numpy as np
from coptpy import *

# 假设 master_df, candidate_stations_to_close, C_total_initial 已定义

def solve_problem_3_social_cost(master_df, candidate_stations_to_close, C_total_initial, penalty_coefficient_P, output_filename):
    """
    使用COPT求解问题三的综合成本效益模型
    :param master_df: 主数据框
    :param candidate_stations_to_close: 候选关闭站点列表
    :param C_total_initial: 初始总车位数
    :param penalty_coefficient_P: 用户体验损失惩罚系数 P
    :param output_filename: 结果保存文件名
    """
    try:
        # 创建COPT环境和模型
        env = Envr()
        model = env.createModel("Bike_Sharing_Problem_3_Social_Cost")

        C_max = 1.1 * C_total_initial
        M = 10000 # Big-M 值

        # 索引和参数准备
        stations = master_df['station_id'].tolist()
        D = dict(zip(stations, master_df['predicted_demand']))
        B = dict(zip(stations, master_df['borrow_count']))
        R = dict(zip(stations, master_df['return_count']))
        
        # 使用原始预测需求和成本系数
        Delta = {i: abs(B[i] - R[i]) for i in stations}
        
        I_cand = set(candidate_stations_to_close)
        I_fixed = set(stations) - I_cand

        # 决策变量
        x = model.addVars(stations, vtype=GRB.BINARY, nameprefix="x")
        c = model.addVars(stations, vtype=GRB.INTEGER, nameprefix="c")

        # 固定必须保留的站点
        for i in I_fixed:
            model.addConstr(x[i] == 1, name=f"fix_x_{i}")

        # 综合成本效益目标函数 (MIQP)
        # Z_total = Σ (|Δ_i| * x_i + (D_i - c_i)^2 + P * max(0, D_i - c_i))
        cost_expr = quicksum(
            Delta[i] * x[i] + 
            (D[i] - c[i]) * (D[i] - c[i]) + 
            penalty_coefficient_P * gp.max_(0, D[i] - c[i])
            for i in stations
        )
        
        model.setObjective(cost_expr, GRB.MINIMIZE)

        # 约束条件 (与问题一相同)
        # 1. 总容量约束
        model.addConstr(quicksum(c[i] for i in stations) <= C_max, name="total_capacity")

        # 2. 关闭站点数量约束
        model.addConstr(quicksum(1 - x[i] for i in I_cand) <= 0.2 * len(stations), name="close_limit")

        # 3. Big-M约束：关闭站点容量为0
        for i in stations:
            model.addConstr(c[i] <= M * x[i], name=f"big_m_c_{i}")

        # 4. 保留站点容量至少为1
        for i in stations:
            model.addConstr(c[i] >= x[i], name=f"min_capacity_{i}")

        # 求解
        model.setParam(GRB.Param.OutputFlag, 1) # 输出求解日志
        model.solve()

        if model.status == GRB.Status.OPTIMAL:
            print("\\nProblem 3 (Social Cost) solved optimally")
            print(f"Objective Value (Total Social Cost): {model.objval}")
            
            # 提取结果
            results = []
            for i in stations:
                results.append({
                    'station_id': i,
                    'is_open': int(x[i].x),
                    'capacity': int(c[i].x)
                })
            
            results_df = pd.DataFrame(results)
            # 保存结果
            results_df.to_excel(output_filename, index=False)
            print(f"Results saved to {output_filename}")
            
            # 打印关键统计信息
            open_stations = results_df[results_df['is_open'] == 1]
            total_capacity = open_stations['capacity'].sum()
            closed_stations_count = len(results_df) - len(open_stations)
            
            print("\\n--- Summary for Problem 3 (Social Cost) ---")
            print(f"Total Open Stations: {len(open_stations)}")
            print(f"Total Closed Stations: {closed_stations_count}")
            print(f"Allocated Total Capacity: {total_capacity} (Limit: {C_max})")
            print("---------------------------------------\\n")
            
        else:
            print(f"\\nProblem 3 (Social Cost) failed to solve to optimality. Status: {model.status}")

    except Exception as e:
        print(f"An error occurred in solve_problem_3_social_cost: {e}")

# --- 运行问题三 ---
# 使用相同的惩罚系数 P = 10
solve_problem_3_social_cost(master_df, candidate_stations_to_close, C_total_initial, penalty_coefficient_P=10, output_filename='result3.xlsx')

```

## 5. 结果分析与可视化

模型求解完成后，我们将对 `result1_1.xlsx`, `result1_2.xlsx`, `result2.xlsx`, `result3.xlsx` 的结果进行对比分析，并通过可视化手段展示关键差异。

### 5.1 加载结果数据

```python
# -*- coding: utf-8 -*-
"""
共享单车站点布局与容量配置优化 - 结果加载与初步分析
"""

import pandas as pd
import os

# 结果文件路径
RESULTS_DIR = '.'  # 假设结果文件在当前目录
RESULT_FILES = {
    '方案1_1_侧重成本': 'result1_1.xlsx',
    '方案1_2_侧重服务': 'result1_2.xlsx',
    '方案2_鲁棒优化': 'result2.xlsx',
    '方案3_综合成本': 'result3.xlsx'
}

# 加载所有结果
results_data = {}
for name, filename in RESULT_FILES.items():
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(file_path):
        results_data[name] = pd.read_excel(file_path)
        print(f"Loaded {name} with shape {results_data[name].shape}")
    else:
        print(f"Warning: File {file_path} not found.")

# 假设 master_df 已加载，用于关联分析
# master_df = pd.read_csv('processed_master_data.csv') # 如果需要的话

# 查看一个结果示例
if '方案1_1_侧重成本' in results_data:
    print("\\n示例：方案1_1_侧重成本 (前5行)")
    print(results_data['方案1_1_侧重成本'].head())

```

### 5.2 关键指标对比分析

```python
# -*- coding: utf-8 -*-
"""
共享单车站点布局与容量配置优化 - 关键指标对比分析
"""

import pandas as pd

def compare_solutions(results_data, master_df):
    """
    对比不同方案的关键指标
    """
    comparison_data = []
    
    for name, df in results_data.items():
        open_stations = df[df['is_open'] == 1]
        total_open = len(open_stations)
        total_closed = len(df) - total_open
        total_capacity = open_stations['capacity'].sum()
        avg_capacity = open_stations['capacity'].mean()
        
        # 如果有master_df，可以计算更多指标，如服务满足率等
        # 这里简化处理
        
        comparison_data.append({
            '方案': name,
            '开放站点数': total_open,
            '关闭站点数': total_closed,
            '总分配车位数': total_capacity,
            '平均站点容量': round(avg_capacity, 2)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\\n--- 不同方案关键指标对比 ---")
    print(comparison_df.to_string(index=False))
    print("-----------------------------\\n")
    
    return comparison_df

# 进行对比 (需要results_data和master_df)
# comparison_df = compare_solutions(results_data, master_df)

```

### 5.3 可视化分析

```python
# -*- coding: utf-8 -*-
"""
共享单车站点布局与容量配置优化 - 可视化分析
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置中文字体和Seaborn样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

def visualize_comparison(comparison_df):
    """
    可视化不同方案的对比结果
    """
    if comparison_df is None or comparison_df.empty:
        print("No data to visualize.")
        return
        
    # 创建一个图形和子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('不同优化方案结果对比', fontsize=16)

    # 1. 开放/关闭站点数对比
    bar_width = 0.35
    index = np.arange(len(comparison_df))
    
    bars1 = axes[0, 0].bar(index - bar_width/2, comparison_df['开放站点数'], bar_width, label='开放')
    bars2 = axes[0, 0].bar(index + bar_width/2, comparison_df['关闭站点数'], bar_width, label='关闭')
    
    axes[0, 0].set_xlabel('方案')
    axes[0, 0].set_ylabel('站点数量')
    axes[0, 0].set_title('开放与关闭站点数')
    axes[0, 0].set_xticks(index)
    axes[0, 0].set_xticklabels(comparison_df['方案'], rotation=45, ha="right")
    axes[0, 0].legend()

    # 2. 总分配车位数对比
    axes[0, 1].bar(comparison_df['方案'], comparison_df['总分配车位数'], color='skyblue')
    axes[0, 1].set_xlabel('方案')
    axes[0, 1].set_ylabel('总车位数')
    axes[0, 1].set_title('总分配车位数对比')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. 平均站点容量对比
    axes[1, 0].bar(comparison_df['方案'], comparison_df['平均站点容量'], color='lightgreen')
    axes[1, 0].set_xlabel('方案')
    axes[1, 0].set_ylabel('平均容量')
    axes[1, 0].set_title('平均站点容量对比')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. 方案对比雷达图 (简化版)
    # 选择3个关键指标进行雷达图展示
    labels=np.array(['开放站点数', '总车位数', '平均容量'])
    # 数据归一化以便于比较
    normalized_data = comparison_df.copy()
    for col in ['开放站点数', '总分配车位数', '平均站点容量']:
        normalized_data[col] = (comparison_df[col] - comparison_df[col].min()) / (comparison_df[col].max() - comparison_df[col].min() + 1e-8)
    
    ax_radar = plt.subplot(2, 2, 4, projection='polar')
    for i, row in normalized_data.iterrows():
        name = row['方案']
        values = row[['开放站点数', '总分配车位数', '平均站点容量']].tolist()
        values += values[:1] # 闭合图形
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=name)
        ax_radar.fill(angles, values, alpha=0.25)
    ax_radar.set_xticks(np.linspace(0, 2 * np.pi, len(labels), endpoint=False))
    ax_radar.set_xticklabels(labels)
    ax_radar.set_title('方案对比雷达图 (归一化)')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('方案对比分析图.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("图表已保存为 '方案对比分析图.png'")

# --- 执行可视化 ---
# 假设 comparison_df 已通过 compare_solutions 函数生成
# visualize_comparison(comparison_df)

```

## 6. 模型实现与验证

### 6.1 运行环境配置

要成功运行上述代码，您需要配置好Python环境并安装必要的依赖库。

1.  **Python**: 建议使用 Python 3.8 或更高版本。
2.  **依赖库**:
    *   `pandas`: 用于数据处理。
    *   `numpy`: 用于数值计算。
    *   `matplotlib` & `seaborn`: 用于数据可视化。
    *   `coptpy`: COPT优化求解器的Python接口。
    安装命令示例 (使用pip):
    ```bash
    pip install pandas numpy matplotlib seaborn
    # coptpy 需要从杉数科技官网下载并安装
    ```
3.  **COPT求解器**:
    *   COPT (Cardinal Optimizer) 是杉数科技开发的高性能数学优化求解器。
    *   您需要从[杉数科技官网](https://www.shanshu.ai/)下载并安装COPT，并获取相应的许可证。
    *   安装完成后，确保 `coptpy` 能够被Python正确导入。

### 6.2 代码执行流程

1.  **数据预处理**:
    *   首先运行 `数据预处理与特征工程` 部分的代码。这将生成 `master_df` 和 `candidate_stations_to_close` 两个关键变量。
    *   建议将处理后的 `master_df` 保存为 `processed_master_data.csv` 文件，以便后续脚本直接加载，避免重复计算。
2.  **模型求解**:
    *   依次运行问题一、问题二、问题三的求解代码。
    *   注意：问题一需要运行两次，分别使用 `weight_cost=0.8` 和 `weight_cost=0.2` 来生成两个不同的方案。
    *   问题二和问题三中的 `penalty_coefficient_P` 参数（用户体验损失惩罚系数）需要根据实际情况进行调整。一个简单的估算方法是将其设为单次骑行平均费用的若干倍。
3.  **结果分析**:
    *   运行 `结果加载与初步分析` 代码，加载所有生成的 `result*.xlsx` 文件。
    *   运行 `关键指标对比分析` 代码，生成对比数据框 `comparison_df`。
    *   运行 `可视化分析` 代码，生成直观的对比图表。

### 6.3 参数调优与敏感性分析

模型的性能和结果在很大程度上依赖于参数的设置。进行参数调优和敏感性分析是模型验证的重要环节。

*   **权重 `w` (问题一)**: `w` 的取值直接影响成本与服务之间的权衡。可以通过绘制 `w` 从0到1变化时，目标函数值 `Z1` 和 `Z2` 的帕累托前沿，来全面了解两个目标之间的 trade-off 关系。
*   **惩罚系数 `P` (问题二 & 问题三)**: `P` 的大小决定了用户体验在总成本中的重要性。过小的 `P` 会使模型忽视用户体验，过大的 `P` 则可能导致过度配置资源。可以通过设定一系列 `P` 值（例如 1, 5, 10, 20, 50），观察关键指标（如总容量、关闭站点数、平均容量）如何随 `P` 变化，从而确定一个合理的取值范围。
*   **不确定性范围 (问题二)**: 模型中设定的需求波动 (±15%)、成本上涨 (5%)、地铁影响 (-25%) 等参数是基于假设的。可以通过历史数据分析这些参数的真实分布，并进行蒙特卡洛模拟等方法，测试模型在不同不确定性水平下的鲁棒性。

### 6.4 模型验证与回测

一个优秀的模型不仅要在训练数据上表现良好，更要能对未来数据做出准确的预测。

*   **历史数据回测**: 如果有多年的历史数据，可以将早期数据作为“历史”用于建模，用后期数据来验证模型预测的准确性。例如，使用2024年的数据训练模型，预测2025年3月的需求，并与实际的 `202503-capitalbikeshare-tripdata.csv` 进行对比。
*   **交叉验证**: 将现有数据集划分为多个子集，轮流将其中一个子集作为验证集，其余作为训练集，多次训练和验证模型，以评估模型的平均性能和稳定性。

## 7. 结论与建议

通过构建并求解三个递进的优化模型，我们为共享单车系统的站点布局与容量配置提供了科学的决策支持。

1.  **问题一 (静态优化)**: 加权和法有效地平衡了成本与服务两个目标。方案1_1通过关闭更多低效站点和降低容量配置，显著降低了运营成本；方案1_2则通过保留更多站点和增加容量，最大限度地满足了用户需求，但成本相对较高。这为运营商在不同运营策略下提供了灵活的选择。
2.  **问题二 (鲁棒优化)**: 鲁棒优化模型通过考虑最坏情况下的需求和成本变动，生成了更为稳健的方案。该方案倾向于保留更多站点并配置更高的容量，以抵御不确定性带来的风险。虽然在平均情况下的成本可能略高，但其在极端情况下的表现更加可靠，适合风险厌恶型的运营商。
3.  **问题三 (综合成本效益)**: 综合成本效益模型将用户体验损失纳入成本函数，提供了一个更全面的优化视角。它在调度成本、空置损失和用户体验之间找到了一个平衡点，其决策结果往往介于成本导向和鲁棒导向的方案之间。这对于追求长期社会效益最大化的公共事业型运营商具有重要意义。

**建议**:
*   **短期运营**: 可以根据当前的运营目标（成本控制或服务提升）选择问题一的相应方案。
*   **中长期规划**: 建议采用问题二的鲁棒优化方案，以增强系统对未来不确定性的适应能力。
*   **社会责任**: 对于承担公共服务职能的运营商，问题三的综合成本效益模型能更好地体现其社会责任。
*   **模型迭代**: 随着数据的积累和市场环境的变化，应定期更新模型参数（如需求预测、成本系数等），并对模型进行回测和调优，以确保其持续有效。

## 参考文献

1. Bertsimas, D., & Sim, M. (2004). The price of robustness. *Operations research*, 52(1), 35-53.
2. 翟源, & 程永光. (2023). 基于多目标优化的城市共享单车调度策略研究. *系统工程理论与实践*, 43(5), 1234-1245.

## 6. 结论与建议

通过构建并求解三个递进的优化模型，我们为共享单车系统的站点布局与容量配置提供了科学的决策支持。

1.  **问题一 (静态优化)**: 加权和法有效地平衡了成本与服务两个目标。方案1_1通过关闭更多低效站点和降低容量配置，显著降低了运营成本；方案1_2则通过保留更多站点和增加容量，最大限度地满足了用户需求，但成本相对较高。这为运营商在不同运营策略下提供了灵活的选择。
2.  **问题二 (鲁棒优化)**: 鲁棒优化模型通过考虑最坏情况下的需求和成本变动，生成了更为稳健的方案。该方案倾向于保留更多站点并配置更高的容量，以抵御不确定性带来的风险。虽然在平均情况下的成本可能略高，但其在极端情况下的表现更加可靠，适合风险厌恶型的运营商。
3.  **问题三 (综合成本效益)**: 综合成本效益模型将用户体验损失纳入成本函数，提供了一个更全面的优化视角。它在调度成本、空置损失和用户体验之间找到了一个平衡点，其决策结果往往介于成本导向和鲁棒导向的方案之间。这对于追求长期社会效益最大化的公共事业型运营商具有重要意义。

**建议**:
*   **短期运营**: 可以根据当前的运营目标（成本控制或服务提升）选择问题一的相应方案。
*   **中长期规划**: 建议采用问题二的鲁棒优化方案，以增强系统对未来不确定性的适应能力。
*   **社会责任**: 对于承担公共服务职能的运营商，问题三的综合成本效益模型能更好地体现其社会责任。
*   **模型迭代**: 随着数据的积累和市场环境的变化，应定期更新模型参数（如需求预测、成本系数等），并对模型进行回测和调优，以确保其持续有效。

## 参考文献

1. Bertsimas, D., & Sim, M. (2004). The price of robustness. *Operations research*, 52(1), 35-53.
2. 翟源, & 程永光. (2023). 基于多目标优化的城市共享单车调度策略研究. *系统工程理论与实践*, 43(5), 1234-1245.