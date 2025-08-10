# -*- coding: utf-8 -*-
"""
共享单车站点布局与容量配置优化 - 使用COPT求解器重新计算
此脚本整合了问题一、问题二和问题三的求解代码，并使用COPT求解器进行重新计算。
"""

import pandas as pd
import numpy as np
import os

# 尝试导入COPT求解器
try:
    from coptpy import *
    print("成功导入COPT求解器。")
    copt_available = True
except ImportError:
    print("警告：未能导入COPT求解器。请确保COPT已正确安装。")
    copt_available = False
    # 如果COPT不可用，我们可以使用其他求解器作为备选，但这超出了当前任务范围。
    # 为了简化，如果COPT不可用，脚本将无法运行优化部分。
    # 我们可以在这里添加一个占位符，或者直接退出。
    exit(1)

# 获取基本目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 查找包含"火花杯"的目录
DATA_DIR = None
for item in os.listdir(BASE_DIR):
    if "火花杯" in item and os.path.isdir(os.path.join(BASE_DIR, item)):
        DATA_DIR = os.path.join(BASE_DIR, item)
        break

if DATA_DIR is None:
    raise FileNotFoundError("未找到包含'火花杯'的目录")

print(f"找到的DATA_DIR: {DATA_DIR}")

# 使用os.path.join构建文件路径
TRIPDATA_FILE = os.path.join(DATA_DIR, '202503-capitalbikeshare-tripdata.csv')
DEMAND_FEATURES_FILE = os.path.join(DATA_DIR, 'demand_features.csv')
METRO_FILE = os.path.join(DATA_DIR, 'metro_coverage.xlsx')

OUTPUT_DIR = BASE_DIR

print(f"TRIPDATA_FILE: {TRIPDATA_FILE}")
print(f"DEMAND_FEATURES_FILE: {DEMAND_FEATURES_FILE}")
print(f"METRO_FILE: {METRO_FILE}")

# 检查文件是否存在
if not os.path.exists(TRIPDATA_FILE):
    raise FileNotFoundError(f"文件未找到: {TRIPDATA_FILE}")
if not os.path.exists(DEMAND_FEATURES_FILE):
    raise FileNotFoundError(f"文件未找到: {DEMAND_FEATURES_FILE}")
if not os.path.exists(METRO_FILE):
    raise FileNotFoundError(f"文件未找到: {METRO_FILE}")

def load_and_preprocess_data():
    """
    加载并预处理数据
    """
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

    print("\nStation activity data (Top 5):")
    print(station_activity_df.head())

    # 2. 加载需求预测数据
    demand_df = pd.read_csv(DEMAND_FEATURES_FILE)
    print("\nDemand features data (Top 5):")
    print(demand_df.head())

    # 3. 加载地铁覆盖数据
    metro_df = pd.read_excel(METRO_FILE)
    print("\nMetro coverage data (Top 5):")
    print(metro_df.head())
    
    # 检查metro_df的列
    print(f"\nMetro coverage data columns: {metro_df.columns.tolist()}")
    
    # 如果metro_df有'station_id'列，则创建一个布尔列'near_metro'
    if 'station_id' in metro_df.columns:
        # 创建一个包含所有地铁站ID的集合
        metro_station_ids = set(metro_df['station_id'].unique())
        # 在demand_df中添加'near_metro'列
        demand_df['near_metro'] = demand_df['station_id'].isin(metro_station_ids)
        print(f"添加了 'near_metro' 列")
    else:
        # 如果没有'station_id'列，则假设所有站点都不在地铁附近
        demand_df['near_metro'] = False
        print(f"未找到 'station_id' 列，'near_metro' 列设置为 False")

    # --- 特征融合与候选站点选择 ---
    
    # 1. 合并所有数据到主数据框
    # 以 demand_df 为基础，因为它包含了所有需要预测的站点
    master_df = demand_df.copy()

    # 合并借还车数据
    master_df = pd.merge(master_df, station_activity_df, on='station_id', how='left').fillna(0)
    
    # 重命名列以避免冲突
    master_df.rename(columns={
        'borrow_count_y': 'borrow_count', 
        'return_count_y': 'return_count'
    }, inplace=True)
    
    # 删除重复的列
    if 'borrow_count_x' in master_df.columns:
        master_df.drop(columns=['borrow_count_x'], inplace=True)
    if 'return_count_x' in master_df.columns:
        master_df.drop(columns=['return_count_x'], inplace=True)

    print("Master DataFrame after merging:")
    print(master_df.head())
    print(f"Shape: {master_df.shape}")
    print(f"Master DataFrame columns: {master_df.columns.tolist()}")

    # 2. 计算初始总车位数
    # 假设初始每个站点30个车位，或使用其他逻辑
    # 更合理的假设是初始总车位数等于所有站点预测需求之和
    C_total_initial = master_df['predicted_demand'].sum() 
    print(f"\n计算初始总车位数 C_total_initial: {C_total_initial}")

    # 3. 选择候选关闭站点 (使用率最低的20%)
    # 按总活动量排序
    master_df_sorted = master_df.sort_values(by='total_activity')
    num_to_close = int(0.2 * len(master_df_sorted))
    candidate_stations_to_close = master_df_sorted.head(num_to_close)['station_id'].tolist()

    print(f"\nIdentified {len(candidate_stations_to_close)} candidate stations for closure (bottom 20% by activity).")
    print("First 10 candidates:", candidate_stations_to_close[:10])
    
    return master_df, candidate_stations_to_close, C_total_initial

def solve_problem_1(master_df, candidate_stations_to_close, C_total_initial, weight_cost, output_filename):
    """
    使用COPT求解问题一的加权和模型
    :param master_df: 主数据框
    :param candidate_stations_to_close: 候选关闭站点列表
    :param C_total_initial: 初始总车位数
    :param weight_cost: 成本目标的权重 w
    :param output_filename: 结果保存文件名
    """
    if not copt_available:
        print("COPT求解器不可用，无法求解问题一。")
        return
        
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
        x = model.addVars(stations, vtype=COPT.BINARY, nameprefix="x")
        c = model.addVars(stations, vtype=COPT.INTEGER, nameprefix="c")
        # 添加辅助变量用于线性化max(0, D-c)
        u = model.addVars(stations, vtype=COPT.CONTINUOUS, nameprefix="u")

        # 固定必须保留的站点
        for i in I_fixed:
            model.addConstr(x[i] == 1, name=f"fix_x_{i}")

        # 线性化约束: u[i] >= D[i] - c[i] 和 u[i] >= 0
        for i in stations:
            model.addConstr(u[i] >= D[i] - c[i], name=f"linearize_u1_{i}")
            model.addConstr(u[i] >= 0, name=f"linearize_u2_{i}")

        # 目标函数组件
        cost_expr = quicksum(Delta[i] * x[i] + (D[i] - c[i]) * (D[i] - c[i]) for i in stations)
        # 使用线性化变量u替代max(0, D-c)
        service_loss_expr = quicksum(u[i] for i in stations)
        
        # 为了加权和，我们需要先求解两个目标的最优值来归一化
        # 这里简化处理，假设已经知道或可以通过两次单独求解得到范围
        # 在实际应用中，应先进行预处理求解以获得 Z1_min, Z1_max, Z2_min, Z2_max
        # 为简化，我们直接使用表达式进行加权 (这在量纲差异大时可能不理想)
        
        # 归一化 (示例值，实际应用中需要预计算)
        # 我们先求解两个目标的最优值来获取范围
        print("计算目标函数的归一化范围...")
        # 解F1最小
        model.setObjective(cost_expr, COPT.MINIMIZE)
        model.setParam(COPT.Param.Logging, 0) # 关闭求解日志
        model.solve()
        if model.status == COPT.OPTIMAL:
            Z1_min = model.objval
            # 固定当前解，求另一个目标的值
            Z2_at_Z1_min = service_loss_expr.getValue()
        else:
            print("无法计算Z1_min")
            Z1_min = 1
            Z2_at_Z1_min = 1
            
        # 解F2最小
        model.setObjective(service_loss_expr, COPT.MINIMIZE)
        model.solve()
        if model.status == COPT.OPTIMAL:
            Z2_min = model.objval
            # 固定当前解，求另一个目标的值
            Z1_at_Z2_min = cost_expr.getValue()
        else:
            print("无法计算Z2_min")
            Z2_min = 1
            Z1_at_Z2_min = 1
            
        Z1_max = Z1_at_Z2_min
        Z2_max = Z2_at_Z1_min
        
        # 避免除以零
        if Z1_max - Z1_min == 0: 
            Z1_max = Z1_min + 1
        if Z2_max - Z2_min == 0:
            Z2_max = Z2_min + 1
            
        print(f"成本范围 (Z1): [{Z1_min:.2f}, {Z1_max:.2f}]")
        print(f"服务损失范围 (Z2): [{Z2_min:.2f}, {Z2_max:.2f}]")

        # 归一化目标函数
        Z1_norm = (cost_expr - Z1_min) / (Z1_max - Z1_min)
        Z2_norm = (service_loss_expr - Z2_min) / (Z2_max - Z2_min)
        
        # 加权目标函数
        model.setObjective(weight_cost * Z1_norm + (1 - weight_cost) * Z2_norm, COPT.MINIMIZE)

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
        model.setParam(COPT.Param.Logging, 1) # 输出求解日志
        print(f"\n开始求解问题一 (权重: 成本={weight_cost}, 服务={1-weight_cost})...")
        model.solve()

        if model.status == COPT.OPTIMAL:
            print(f"\nProblem 1 solved optimally with weight_cost={weight_cost}")
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
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            results_df.to_excel(output_path, index=False)
            print(f"Results saved to {output_path}")
            
            # 打印一些关键统计信息
            open_stations = results_df[results_df['is_open'] == 1]
            total_capacity = open_stations['capacity'].sum()
            closed_stations_count = len(results_df) - len(open_stations)
            
            print(f"\n--- Summary for weight_cost={weight_cost} ---")
            print(f"Total Open Stations: {len(open_stations)}")
            print(f"Total Closed Stations: {closed_stations_count}")
            print(f"Allocated Total Capacity: {total_capacity} (Limit: {C_max})")
            
            # 计算实际的 Z1 和 Z2 值
            Z1_actual = sum(Delta[i] * results_df.loc[results_df['station_id']==i, 'is_open'].iloc[0] + 
                            (D[i] - results_df.loc[results_df['station_id']==i, 'capacity'].iloc[0])**2 
                            for i in stations)
            # 对于Z2_actual，我们使用线性化变量的值
            Z2_actual = sum(max(0, D[i] - results_df.loc[results_df['station_id']==i, 'capacity'].iloc[0]) 
                            for i in stations)
            print(f"Actual Cost (Z1): {Z1_actual}")
            print(f"Actual Service Loss (Z2): {Z2_actual}")
            print("---------------------------------------\n")
            
        else:
            print(f"\nProblem 1 failed to solve to optimality with weight_cost={weight_cost}. Status: {model.status}")

    except Exception as e:
        print(f"An error occurred in solve_problem_1: {e}")


def solve_problem_2_robust(master_df, candidate_stations_to_close, C_total_initial, penalty_coefficient_P, output_filename):
    """
    使用COPT求解问题二的鲁棒优化模型
    :param master_df: 主数据框
    :param candidate_stations_to_close: 候选关闭站点列表
    :param C_total_initial: 初始总车位数
    :param penalty_coefficient_P: 用户体验损失惩罚系数 P
    :param output_filename: 结果保存文件名
    """
    if not copt_available:
        print("COPT求解器不可用，无法求解问题二。")
        return
        
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
        x = model.addVars(stations, vtype=COPT.BINARY, nameprefix="x")
        c = model.addVars(stations, vtype=COPT.INTEGER, nameprefix="c")

        # 固定必须保留的站点
        for i in I_fixed:
            model.addConstr(x[i] == 1, name=f"fix_x_{i}")

        # 鲁棒优化目标函数 (MIQP)
        # Z_robust = Σ (1.05 * |Δ_i| * x_i + (D_i_robust - c_i)^2 + P * max(0, D_i_robust - c_i))
        # 使用辅助变量线性化 max(0, D_robust - c_i)
        u = model.addVars(stations, vtype=COPT.CONTINUOUS, nameprefix="u")
        for i in stations:
            model.addConstr(u[i] >= D_robust[i] - c[i], name=f"linearize_u1_{i}")
            model.addConstr(u[i] >= 0, name=f"linearize_u2_{i}")
        
        cost_expr = quicksum(
            Delta_robust[i] * x[i] + 
            (D_robust[i] - c[i]) * (D_robust[i] - c[i]) + 
            penalty_coefficient_P * u[i]
            for i in stations
        )
        
        model.setObjective(cost_expr, COPT.MINIMIZE)

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
        model.setParam(COPT.Param.Logging, 1) # 输出求解日志
        print(f"\n开始求解问题二 (鲁棒优化)...")
        model.solve()

        if model.status == COPT.OPTIMAL:
            print("\nProblem 2 (Robust) solved optimally")
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
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            results_df.to_excel(output_path, index=False)
            print(f"Results saved to {output_path}")
            
            # 打印关键统计信息
            open_stations = results_df[results_df['is_open'] == 1]
            total_capacity = open_stations['capacity'].sum()
            closed_stations_count = len(results_df) - len(open_stations)
            
            print("\n--- Summary for Problem 2 (Robust) ---")
            print(f"Total Open Stations: {len(open_stations)}")
            print(f"Total Closed Stations: {closed_stations_count}")
            print(f"Allocated Total Capacity: {total_capacity} (Limit: {C_max})")
            print("---------------------------------------\n")
            
        else:
            print(f"\nProblem 2 (Robust) failed to solve to optimality. Status: {model.status}")

    except Exception as e:
        print(f"An error occurred in solve_problem_2_robust: {e}")


def solve_problem_3_social_cost(master_df, candidate_stations_to_close, C_total_initial, penalty_coefficient_P, output_filename):
    """
    使用COPT求解问题三的综合成本效益模型
    :param master_df: 主数据框
    :param candidate_stations_to_close: 候选关闭站点列表
    :param C_total_initial: 初始总车位数
    :param penalty_coefficient_P: 用户体验损失惩罚系数 P
    :param output_filename: 结果保存文件名
    """
    if not copt_available:
        print("COPT求解器不可用，无法求解问题三。")
        return
        
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
        x = model.addVars(stations, vtype=COPT.BINARY, nameprefix="x")
        c = model.addVars(stations, vtype=COPT.INTEGER, nameprefix="c")
        # 使用辅助变量线性化 max(0, D - c)
        u = model.addVars(stations, vtype=COPT.CONTINUOUS, nameprefix="u")

        # 固定必须保留的站点
        for i in I_fixed:
            model.addConstr(x[i] == 1, name=f"fix_x_{i}")

        # 线性化约束
        for i in stations:
            model.addConstr(u[i] >= D[i] - c[i], name=f"linearize_u1_{i}")
            model.addConstr(u[i] >= 0, name=f"linearize_u2_{i}")

        # 综合成本效益目标函数 (MIQP)
        # Z_total = Σ (|Δ_i| * x_i + (D_i - c_i)^2 + P * max(0, D_i - c_i))
        cost_expr = quicksum(
            Delta[i] * x[i] + 
            (D[i] - c[i]) * (D[i] - c[i]) + 
            penalty_coefficient_P * u[i]
            for i in stations
        )
        
        model.setObjective(cost_expr, COPT.MINIMIZE)

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
        model.setParam(COPT.Param.Logging, 1) # 输出求解日志
        print(f"\n开始求解问题三 (综合成本效益)...")
        model.solve()

        if model.status == COPT.OPTIMAL:
            print("\nProblem 3 (Social Cost) solved optimally")
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
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            results_df.to_excel(output_path, index=False)
            print(f"Results saved to {output_path}")
            
            # 打印关键统计信息
            open_stations = results_df[results_df['is_open'] == 1]
            total_capacity = open_stations['capacity'].sum()
            closed_stations_count = len(results_df) - len(open_stations)
            
            print("\n--- Summary for Problem 3 (Social Cost) ---")
            print(f"Total Open Stations: {len(open_stations)}")
            print(f"Total Closed Stations: {closed_stations_count}")
            print(f"Allocated Total Capacity: {total_capacity} (Limit: {C_max})")
            print("---------------------------------------\n")
            
        else:
            print(f"\nProblem 3 (Social Cost) failed to solve to optimality. Status: {model.status}")

    except Exception as e:
        print(f"An error occurred in solve_problem_3_social_cost: {e}")


def main():
    """
    主函数
    """
    print("开始重新使用COPT求解器计算共享单车优化问题...")
    
    # 1. 加载并预处理数据
    master_df, candidate_stations_to_close, C_total_initial = load_and_preprocess_data()
    
    # 2. 求解问题一的两个方案
    # 方案 1_1: 侧重成本 (高权重给成本)
    solve_problem_1(master_df, candidate_stations_to_close, C_total_initial, weight_cost=0.8, output_filename='result1_1_copt.xlsx')

    # 方案 1_2: 侧重服务 (低权重给成本)
    solve_problem_1(master_df, candidate_stations_to_close, C_total_initial, weight_cost=0.2, output_filename='result1_2_copt.xlsx')
    
    # 3. 求解问题二
    # 假设用户体验损失惩罚系数 P = 10 (需要根据实际情况调整)
    solve_problem_2_robust(master_df, candidate_stations_to_close, C_total_initial, penalty_coefficient_P=10, output_filename='result2_copt.xlsx')
    
    # 4. 求解问题三
    # 使用相同的惩罚系数 P = 10
    solve_problem_3_social_cost(master_df, candidate_stations_to_close, C_total_initial, penalty_coefficient_P=10, output_filename='result3_copt.xlsx')
    
    print("\n所有问题求解完成。结果已保存。")

if __name__ == "__main__":
    main()