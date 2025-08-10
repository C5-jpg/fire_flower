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
    加载原始数据，进行预处理，计算模型所需参数，并返回一个主数据框。
    
    参数:
    base_path (str): 存放所有附件的文件夹路径。
    
    返回:
    pandas.DataFrame: 包含所有站点信息的 master_df。
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

    # 3. 数据清洗和准备
    # 移除站点ID为空的记录
    df_trip.dropna(subset=['start_station_id', 'end_station_id'], inplace=True)
    # 统一站点ID为整数
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
    # 借还差值
    master_df['borrow_return_diff'] = master_df['borrow_count'] - master_df['return_count']
    
    # 调度成本系数 (借还差值的约数和)
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

    # 站点利用率 (总周转量)
    master_df['turnover'] = master_df['borrow_count'] + master_df['return_count']
    
    # 确定可关闭的站点 (利用率最低的20%)
    n_stations = len(master_df)
    n_to_close_candidates = int(0.2 * n_stations)
    low_utilization_stations = master_df.nsmallest(n_to_close_candidates, 'turnover')['station_id']
    master_df['is_candidate_to_close'] = master_df['station_id'].isin(low_utilization_stations)

    # 7. 估算初始总容量并计算容量上限
    # 假设初始容量等于预测需求量，这是一个常见的合理假设
    initial_total_capacity = master_df['predicted_demand'].sum()
    capacity_upper_bound = 1.1 * initial_total_capacity
    print(f"估算初始总容量: {initial_total_capacity:.0f}")
    print(f"优化后总容量上限: {capacity_upper_bound:.0f}")
    
    # 将容量上限作为一个属性附加到DataFrame上，方便传递
    master_df.capacity_upper_bound = capacity_upper_bound
    
    print("--- 数据预处理完成 ---")
    return master_df.set_index('station_id')


# =============================================================================
# 第二部分: 优化建模与求解
# =============================================================================

def solve_optimization_model(master_df, weight_cost, result_filename):
    """
    构建并求解MIQP模型。
    
    参数:
    master_df (pandas.DataFrame): 预处理后的主数据框。
    weight_cost (float): 运营成本目标的权重 (0到1之间)。
    result_filename (str): 保存结果的Excel文件名。
    """
    print(f"\n--- 开始求解优化模型 (成本权重: {weight_cost}) ---")
    
    # 1. 提取参数
    stations = master_df.index.tolist()
    I = range(len(stations))
    station_map = dict(zip(I, stations))

    D = master_df['predicted_demand'].to_dict()
    S = master_df['scheduling_cost_coeff'].to_dict()
    is_candidate = master_df['is_candidate_to_close'].to_dict()
    
    C_max = master_df.capacity_upper_bound
    max_close_count = int(0.2 * len(stations))
    M = master_df['predicted_demand'].max() * 1.5 # Big-M

    # 2. 创建COPT环境和模型
    env = cp.Envr()
    model = env.createModel(f"bike_opt_w{weight_cost}")

    # 3. 添加决策变量
    x = model.addVars(stations, vtype=COPT.BINARY, nameprefix="x") # 站点是否保留
    c = model.addVars(stations, vtype=COPT.INTEGER, nameprefix="c") # 站点容量
    
    # 辅助变量用于线性化max(0, D-c)
    service_loss_vars = model.addVars(stations, vtype=COPT.CONTINUOUS, nameprefix="u")

    # 4. 添加约束
    # 总容量约束
    model.addConstr(cp.quicksum(c[i] for i in stations) <= C_max)
    
    # 站点关闭数量约束
    candidate_stations = [i for i in stations if is_candidate[i]]
    model.addConstr(cp.quicksum(1 - x[i] for i in candidate_stations) <= max_close_count)

    # 固定站点必须保留
    fixed_stations = [i for i in stations if not is_candidate[i]]
    model.addConstrs((x[i] == 1 for i in fixed_stations))

    # 容量与站点状态关联 (Big-M)
    model.addConstrs((c[i] <= M * x[i] for i in stations))
    model.addConstrs((c[i] >= x[i] for i in stations)) # 保留的站点容量至少为1
    
    # 线性化服务损失
    model.addConstrs((service_loss_vars[i] >= D[i] - c[i] for i in stations))
    model.addConstrs((service_loss_vars[i] >= 0 for i in stations))

    # 5. 定义目标函数
    # 目标F1: 运营成本 (调度成本 + 空置损失)
    cost_scheduling = cp.quicksum(S[i] * x[i] for i in stations)
    cost_vacancy = cp.quicksum((D[i] - c[i]) * (D[i] - c[i]) for i in stations)
    F1_cost = cost_scheduling + cost_vacancy

    # 目标F2: 服务损失
    F2_service_loss = cp.quicksum(service_loss_vars[i] for i in stations)

    # 6. 归一化处理 (重要步骤)
    # 为了得到归一化的范围，我们需要先单独求解每个目标
    print("计算目标函数的归一化范围...")
    # 解F1最小
    model.setObjective(F1_cost, COPT.MINIMIZE)
    model.solve()
    F1_min = model.objval
    F2_at_F1_min = F2_service_loss.getValue()

    # 解F2最小
    model.setObjective(F2_service_loss, COPT.MINIMIZE)
    model.solve()
    F2_min = model.objval
    F1_at_F2_min = F1_cost.getValue()

    F1_max = F1_at_F2_min
    F2_max = F2_at_F1_min
    
    print(f"成本范围 (F1): [{F1_min:.2f}, {F1_max:.2f}]")
    print(f"服务损失范围 (F2): [{F2_min:.2f}, {F2_max:.2f}]")

    # 7. 设置加权和目标函数
    norm_F1 = (F1_cost - F1_min) / (F1_max - F1_min)
    norm_F2 = (F2_service_loss - F2_min) / (F2_max - F2_min)
    
    weight_service = 1 - weight_cost
    
    model.setObjective(weight_cost * norm_F1 + weight_service * norm_F2, COPT.MINIMIZE)

    # 8. 求解最终模型
    print("求解加权和模型...")
    model.solve()

    # 9. 分析和保存结果
    if model.status == COPT.OPTIMAL:
        print("找到最优解!")
        solution = []
        for i in stations:
            is_open = int(round(x[i].x))
            capacity = int(round(c[i].x)) if is_open else 0
            solution.append({
                'station_id': i,
                'is_retained': is_open, # 1表示保留, 0表示关闭
                'capacity': capacity
            })
        
        result_df = pd.DataFrame(solution)
        
        # 获取项目根目录，并将结果保存在那里
        output_path = os.path.join(os.path.dirname(base_path), result_filename)
        result_df.to_excel(output_path, index=False)
        print(f"结果已成功保存到: {output_path}")
        print(result_df.head())
    else:
        print(f"未能找到最优解。状态码: {model.status}")


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == '__main__':
    # 推荐使用原始字符串和绝对路径，避免路径转义和中文符号问题
    base_folder_path = r"C:\Users\hk\fire_flower\2025年度“火花杯”数学建模精英联赛-B题-附件"

    # 步骤一：数据预处理
    master_data = preprocess_data(base_folder_path)

    # 步骤二：求解并生成两个结果文件
    # 方案1: 成本优先 (result1_1.xlsx)
    solve_optimization_model(master_data, weight_cost=0.8, result_filename="result1_1.xlsx")

    # 方案2: 服务优先 (result1_2.xlsx)
    solve_optimization_model(master_data, weight_cost=0.2, result_filename="result1_2.xlsx")

