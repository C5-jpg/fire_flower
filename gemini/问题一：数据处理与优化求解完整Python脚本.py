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

    # 检查并删除 'demand_features.csv' 中已存在的计数列，以避免合并时产生列名冲突
    if 'borrow_count' in df_demand.columns and 'return_count' in df_demand.columns:
        print("发现 'demand_features.csv' 中已存在借还计数，将予以忽略并基于骑行数据重新计算。")
        df_demand = df_demand.drop(columns=['borrow_count', 'return_count'])

    # 3. 数据清洗和准备
    df_trip.dropna(subset=['start_station_id', 'end_station_id'], inplace=True)
    df_trip['start_station_id'] = df_trip['start_station_id'].astype(int)
    df_trip['end_station_id'] = df_trip['end_station_id'].astype(int)
    df_demand.dropna(subset=['station_id'], inplace=True)
    df_demand['station_id'] = df_demand['station_id'].astype(int)

    # 4. 计算每个站点的借车和还车次数
    print("计算各站点借车、还车次数...")
    borrow_counts = df_trip['start_station_id'].value_counts().reset_index()
    borrow_counts.columns = ['station_id', 'borrow_count']

    return_counts = df_trip['end_station_id'].value_counts().reset_index()
    return_counts.columns = ['station_id', 'return_count']

    # 5. 合并借还数据到站点特征数据中
    master_df = pd.merge(df_demand, borrow_counts, on='station_id', how='left')
    master_df = pd.merge(master_df, return_counts, on='station_id', how='left')
    master_df[['borrow_count', 'return_count']] = master_df[['borrow_count', 'return_count']].fillna(0)

    # 6. 计算模型所需参数
    print("计算模型所需的核心参数...")
    master_df['borrow_return_diff'] = master_df['borrow_count'] - master_df['return_count']
    
    def sum_of_divisors(n):
        n = abs(int(n))
        if n == 0:
            return 0
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

    # 7. 估算初始总容量并计算容量上限
    initial_total_capacity = master_df['predicted_demand'].sum()
    capacity_upper_bound = 1.1 * initial_total_capacity
    print(f"估算初始总容量: {initial_total_capacity:.0f}")
    print(f"优化后总容量上限: {capacity_upper_bound:.0f}")
    
    print("--- 数据预处理完成 ---")
    return master_df.set_index('station_id'), capacity_upper_bound


# =============================================================================
# 第二部分: 优化建模与求解
# =============================================================================

def solve_optimization_model(base_path, master_df, C_max, weight_cost, result_filename):
    """
    构建并求解MIQP模型。
    
    参数:
    base_path (str): 存放所有附件的文件夹路径。
    master_df (pandas.DataFrame): 预处理后的主数据框。
    C_max (float): 总容量上限。
    weight_cost (float): 运营成本目标的权重 (0到1之间)。
    result_filename (str): 保存结果的Excel文件名。
    """
    print(f"\n--- 开始求解优化模型 (成本权重: {weight_cost}) ---")
    
    # 1. 提取参数
    stations = master_df.index.tolist()
    D = master_df['predicted_demand'].to_dict()
    S = master_df['scheduling_cost_coeff'].to_dict()
    is_candidate = master_df['is_candidate_to_close'].to_dict()
    
    max_close_count = int(0.2 * len(stations))
    M = master_df['predicted_demand'].max() * 1.5

    # 2. 创建COPT环境和模型
    env = cp.Envr()
    model = env.createModel(f"bike_opt_w{weight_cost}")

    # --- 新增代码：启用GPU加速 ---
    # 1. 设置根节点求解算法为内点法 (Barrier)，这是启用GPU的前提
    model.setParam("RootLpMethod", 3)
    # 2. 启用GPU加速 (1=启用, 0=禁用)
    model.setParam("Gpu", 1)
    print("GPU加速已启用。")
    # --- 新增代码结束 ---

    # 3. 添加决策变量
    x = model.addVars(stations, vtype=COPT.BINARY, nameprefix="x")
    c = model.addVars(stations, vtype=COPT.INTEGER, nameprefix="c")
    service_loss_vars = model.addVars(stations, vtype=COPT.CONTINUOUS, nameprefix="u")

    # 4. 添加约束
    model.addConstr(cp.quicksum(c[i] for i in stations) <= C_max, name="total_capacity")
    
    candidate_stations = [i for i in stations if is_candidate[i]]
    model.addConstr(cp.quicksum(1 - x[i] for i in candidate_stations) <= max_close_count, name="max_close")

    fixed_stations = [i for i in stations if not is_candidate[i]]
    model.addConstrs((x[i] == 1 for i in fixed_stations))

    model.addConstrs((c[i] <= M * x[i] for i in stations), nameprefix="cap_link_upper")
    model.addConstrs((c[i] >= x[i] for i in stations), nameprefix="cap_link_lower")
    
    model.addConstrs((service_loss_vars[i] >= D[i] - c[i] for i in stations), nameprefix="service_loss_linearize")
    model.addConstrs((service_loss_vars[i] >= 0 for i in stations), nameprefix="service_loss_non_negative")

    # 5. 定义目标函数
    cost_scheduling = cp.quicksum(S[i] * x[i] for i in stations)
    cost_vacancy = cp.quicksum((D[i] - c[i]) * (D[i] - c[i]) for i in stations)
    F1_cost = cost_scheduling + cost_vacancy

    F2_service_loss = cp.quicksum(service_loss_vars[i] for i in stations)

    # 6. 归一化处理
    print("计算目标函数的归一化范围...")
    model.setObjective(F1_cost, COPT.MINIMIZE)
    model.solve()
    F1_min = model.objval if model.status == COPT.OPTIMAL else 0

    model.setObjective(F2_service_loss, COPT.MINIMIZE)
    model.solve()
    F2_min = model.objval if model.status == COPT.OPTIMAL else 0
    
    # 为了简化，我们用一个简单的方法估算范围，避免求解最大化问题
    F1_max = F1_min * 100 
    F2_max = F2_min * 100
    
    if (F1_max - F1_min) == 0 or (F2_max - F2_min) == 0:
        print("警告: 目标函数范围为零，无法进行归一化，将使用原始目标。")
        model.setObjective(weight_cost * F1_cost + (1-weight_cost) * F2_service_loss, COPT.MINIMIZE)
    else:
        norm_F1 = (F1_cost - F1_min) / (F1_max - F1_min)
        norm_F2 = (F2_service_loss - F2_min) / (F2_max - F2_min)
        model.setObjective(weight_cost * norm_F1 + (1 - weight_cost) * norm_F2, COPT.MINIMIZE)

    # 8. 求解最终模型
    print("求解加权和模型...")
    model.solve()

    # 9. 分析和保存结果
    if model.status == COPT.OPTIMAL or model.status == COPT.FEASIBLE:
        print("找到解!")
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
        
        output_path = os.path.join(base_path, result_filename)
        result_df.to_excel(output_path, index=False)
        print(f"结果已成功保存到: {output_path}")
        print(result_df.head())
    else:
        print(f"未能找到解。状态码: {model.status}")


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == '__main__':
    base_folder_path = "C:/Users/hk/fire_flower/attachment"

    master_data, C_max_val = preprocess_data(base_folder_path)

    # 方案1: 成本优先 (result1_1.xlsx)
    solve_optimization_model(base_folder_path, master_data, C_max_val, weight_cost=0.8, result_filename="result1_1.xlsx")

    # 方案2: 服务优先 (result1_2.xlsx)
    solve_optimization_model(base_folder_path, master_data, C_max_val, weight_cost=0.2, result_filename="result1_2.xlsx")
