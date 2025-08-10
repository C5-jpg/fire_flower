import pandas as pd
import numpy as np
import coptpy as cp
from coptpy import COPT
import os

# =============================================================================
# 第一部分: 数据预处理与参数计算
# =============================================================================

def preprocess_data(base_path):
    """
    加载原始数据，进行预处理，计算模型所需参数。
    
    参数:
    base_path (str): 存放所有附件的文件夹路径。
    
    返回:
    tuple: (pandas.DataFrame, float) 包含站点信息的数据框和总容量上限。
    """
    print("--- 开始数据预处理 ---")

    # 1. 定义文件路径
    trip_data_path = os.path.join(base_path, "202503-capitalbikeshare-tripdata.csv")
    demand_features_path = os.path.join(base_path, "demand_features.csv")

    # 2. 加载数据
    print(f"加载骑行数据: {trip_data_path}")
    df_trip = pd.read_csv(trip_data_path)
    print(f"加载站点特征数据: {demand_features_path}")
    df_demand = pd.read_csv(demand_features_path)

    # 3. 检查并移除 demand_features.csv 中可能存在的旧计数，避免合并冲突
    if 'borrow_count' in df_demand.columns and 'return_count' in df_demand.columns:
        print("发现 'demand_features.csv' 中已存在借还计数，将予以忽略并重新计算。")
        df_demand = df_demand.drop(columns=['borrow_count', 'return_count'])

    # 4. 数据清洗和准备
    df_trip.dropna(subset=['start_station_id', 'end_station_id'], inplace=True)
    df_trip['start_station_id'] = df_trip['start_station_id'].astype(int)
    df_trip['end_station_id'] = df_trip['end_station_id'].astype(int)
    df_demand.dropna(subset=['station_id'], inplace=True)
    df_demand['station_id'] = df_demand['station_id'].astype(int)

    # 5. 计算每个站点的借车和还车次数
    print("计算各站点借车、还车次数...")
    borrow_counts = df_trip['start_station_id'].value_counts().reset_index()
    borrow_counts.columns = ['station_id', 'borrow_count']

    return_counts = df_trip['end_station_id'].value_counts().reset_index()
    return_counts.columns = ['station_id', 'return_count']

    # 6. 合并数据
    master_df = pd.merge(df_demand, borrow_counts, on='station_id', how='left')
    master_df = pd.merge(master_df, return_counts, on='station_id', how='left')
    master_df[['borrow_count', 'return_count']] = master_df[['borrow_count', 'return_count']].fillna(0)

    # 7. 计算模型所需参数
    print("计算模型所需的核心参数...")
    master_df['borrow_return_diff'] = master_df['borrow_count'] - master_df['return_count']
    
    def sum_of_divisors(n):
        n = abs(int(n))
        if n == 0: return 0
        div_sum = 0
        for i in range(1, int(np.sqrt(n)) + 1):
            if n % i == 0:
                div_sum += i
                if i*i != n:
                    div_sum += n // i
        return div_sum

    master_df['scheduling_cost_coeff'] = master_df['borrow_return_diff'].apply(sum_of_divisors)
    master_df['turnover'] = master_df['borrow_count'] + master_df['return_count']
    
    n_stations = len(master_df)
    n_to_close_candidates = int(0.2 * n_stations)
    low_utilization_stations = master_df.nsmallest(n_to_close_candidates, 'turnover')['station_id']
    master_df['is_candidate_to_close'] = master_df['station_id'].isin(low_utilization_stations)

    # 8. 估算容量上限
    initial_total_capacity = master_df['predicted_demand'].sum()
    capacity_upper_bound = 1.1 * initial_total_capacity
    
    print("--- 数据预处理完成 ---")
    return master_df.set_index('station_id'), capacity_upper_bound


# =============================================================================
# 第二部分: 问题三模型求解
# =============================================================================

def solve_problem_3(base_path, master_df, C_max):
    """
    构建并求解问题三的综合成本效益模型 (MIQP)。

    参数:
    base_path (str): 存放所有附件的文件夹路径。
    master_df (pandas.DataFrame): 预处理好的主数据框。
    C_max (float): 总容量上限。
    """
    
    result_filename = os.path.join(base_path, "result3_optimized.xlsx")
    
    print("\n--- 开始求解问题三：综合成本效益模型 ---")
    
    # 1. 定义模型参数
    P_penalty_coefficient = 15.0
    print(f"用户体验损失惩罚系数 (P) 设置为: {P_penalty_coefficient}")

    stations = master_df.index.tolist()
    D = master_df['predicted_demand'].to_dict()
    S = master_df['scheduling_cost_coeff'].to_dict()
    is_candidate = master_df['is_candidate_to_close'].to_dict()
    
    max_close_count = int(0.2 * len(stations))
    M = master_df['predicted_demand'].max() * 1.5

    # 2. 创建COPT环境和模型
    env = cp.Envr()
    model = env.createModel("bike_optimization_p3")

    # 3. 添加决策变量
    x = model.addVars(stations, vtype=COPT.BINARY, nameprefix="x")
    c = model.addVars(stations, vtype=COPT.INTEGER, nameprefix="c")
    u = model.addVars(stations, vtype=COPT.CONTINUOUS, nameprefix="u")

    # 4. 添加约束
    model.addConstr(cp.quicksum(c[i] for i in stations) <= C_max, name="total_capacity")
    
    candidate_stations = [i for i in stations if is_candidate[i]]
    model.addConstr(cp.quicksum(1 - x[i] for i in candidate_stations) <= max_close_count, name="max_close")

    fixed_stations = [i for i in stations if not is_candidate[i]]
    # **【修复】** 移除了不支持的 'name' 参数
    model.addConstrs((x[i] == 1 for i in fixed_stations))

    model.addConstrs((c[i] <= M * x[i] for i in stations), nameprefix="cap_link_upper")
    model.addConstrs((c[i] >= x[i] for i in stations), nameprefix="cap_link_lower")
    
    model.addConstrs((u[i] >= D[i] - c[i] for i in stations), nameprefix="user_loss_linearize")
    model.addConstrs((u[i] >= 0 for i in stations), nameprefix="user_loss_non_negative")

    # 5. 定义并设置综合成本目标函数
    cost_dispatch = cp.quicksum(S[i] * x[i] for i in stations)
    cost_vacancy = cp.quicksum((D[i] - c[i]) * (D[i] - c[i]) for i in stations)
    cost_user_exp = cp.quicksum(P_penalty_coefficient * u[i] for i in stations)
    
    total_social_cost = cost_dispatch + cost_vacancy + cost_user_exp
    
    model.setObjective(total_social_cost, COPT.MINIMIZE)
    
    print("模型构建完成，开始使用COPT求解...")
    
    # 6. 求解模型
    model.solve()

    # 7. 分析和保存结果
    if model.status == COPT.OPTIMAL or model.status == COPT.FEASIBLE:
        status_msg = "找到最优解!" if model.status == COPT.OPTIMAL else "在规定时间内找到一个可行解。"
        print(status_msg)
            
        solution = []
        for i in stations:
            is_open = int(round(x[i].x))
            capacity = int(round(c[i].x)) if is_open else 0
            solution.append({
                'station_id': i,
                'is_retained': is_open,
                'capacity': capacity
            })
        
        result_df = pd.DataFrame(solution)
        
        result_df.to_excel(result_filename, index=False)
        print(f"问题三结果已成功保存到: {result_filename}")
        print(result_df.head())
    else:
        print(f"未能找到解。状态码: {model.status}")

# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == '__main__':
    # 路径已更新为您指定的简洁路径
    # 注意: 在Python代码中，路径最好使用正斜杠 "/"
    base_folder_path = "C:/Users/hk/fire_flower/attachment"
    
    # 步骤一：同时接收数据框和容量上限
    master_data, C_max_val = preprocess_data(base_folder_path)
    
    # 步骤二：将容量上限作为参数传入求解函数
    solve_problem_3(base_folder_path, master_data, C_max_val)
