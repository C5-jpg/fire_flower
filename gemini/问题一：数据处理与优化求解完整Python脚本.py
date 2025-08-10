# =============================================================================
# bike_optimization_solution_cn.py
#
# 描述:
# 解决一个共享单车系统的多目标优化问题。
# 该模型旨在：
#  1. 最小化运营成本和空置成本。
#  2. 最小化服务损失（未满足的需求）。
# 模型采用加权和方法来处理两个目标，并配置为使用 COPT 优化器进行 GPU 加速。
#
# 依赖库: pandas, numpy, coptpy
# 请确保已安装 COPT 并拥有有效的许可证。
#
# 作者: Gemini AI (基于用户的初始脚本)
# 日期: 2025-08-10
# =============================================================================

import pandas as pd
import numpy as np
import coptpy as cp
from coptpy import COPT
import os
import sys

# =============================================================================
# 第一部分: 数据预处理与参数计算
# =============================================================================

def preprocess_data(base_path):
    """
    加载原始数据，进行预处理，并计算优化模型所需的参数。

    参数:
        base_path (str): 存放所有数据文件的文件夹路径。

    返回:
        tuple: 一个元组，包含:
            - pandas.DataFrame: 以 'station_id' 为索引的站点信息数据框。
            - float: 计算得出的系统总容量上限。
    """
    print("--- 开始数据预处理 ---")

    # 1. 定义文件路径
    trip_data_path = os.path.join(base_path, "202503-capitalbikeshare-tripdata.csv")
    demand_features_path = os.path.join(base_path, "demand_features.csv")

    # 检查文件是否存在
    if not os.path.exists(trip_data_path) or not os.path.exists(demand_features_path):
        print(f"错误: 在 '{base_path}' 中未找到数据文件。请检查路径。", file=sys.stderr)
        sys.exit(1)

    # 2. 加载数据
    print(f"加载骑行数据: {trip_data_path}")
    df_trip = pd.read_csv(trip_data_path)
    print(f"加载站点特征数据: {demand_features_path}")
    df_demand = pd.read_csv(demand_features_path)

    # --- BUG 修复 ---
    # 检查并删除 'demand_features.csv' 中可能已存在的计数列，以避免合并时产生列名冲突
    cols_to_check = ['borrow_count', 'return_count']
    existing_cols_to_drop = [col for col in cols_to_check if col in df_demand.columns]
    if existing_cols_to_drop:
        print(f"发现 'demand_features.csv' 中已存在以下列: {existing_cols_to_drop}。将予以忽略并基于骑行数据重新计算。")
        df_demand = df_demand.drop(columns=existing_cols_to_drop)
    # --- 修复结束 ---

    # 3. 数据清洗和准备
    print("正在清洗和准备数据...")
    df_trip.dropna(subset=['start_station_id', 'end_station_id'], inplace=True)
    df_trip['start_station_id'] = df_trip['start_station_id'].astype(int)
    df_trip['end_station_id'] = df_trip['end_station_id'].astype(int)
    df_demand.dropna(subset=['station_id'], inplace=True)
    df_demand['station_id'] = df_demand['station_id'].astype(int)

    # 4. 计算每个站点的借车和还车次数
    print("正在计算各站点的借还车次数...")
    borrow_counts = df_trip['start_station_id'].value_counts().reset_index()
    borrow_counts.columns = ['station_id', 'borrow_count']

    return_counts = df_trip['end_station_id'].value_counts().reset_index()
    return_counts.columns = ['station_id', 'return_count']

    # 5. 将借还车次数合并到主站点数据框中
    master_df = pd.merge(df_demand, borrow_counts, on='station_id', how='left')
    master_df = pd.merge(master_df, return_counts, on='station_id', how='left')
    
    # 用0填充可能因左连接产生的NaN值
    master_df[['borrow_count', 'return_count']] = master_df[['borrow_count', 'return_count']].fillna(0)

    # 6. 计算模型参数
    print("正在计算优化模型所需的核心参数...")
    master_df['borrow_return_diff'] = master_df['borrow_count'] - master_df['return_count']

    def sum_of_divisors(n):
        # 计算一个数所有正因子的和
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

    # 调度成本系数与净流量差值的因子和成正比
    master_df['scheduling_cost_coeff'] = master_df['borrow_return_diff'].apply(sum_of_divisors)
    master_df['turnover'] = master_df['borrow_count'] + master_df['return_count'] # 总周转量

    n_stations = len(master_df)
    n_to_close_candidates = int(0.2 * n_stations)
    low_utilization_stations = master_df.nsmallest(n_to_close_candidates, 'turnover')['station_id']
    master_df['is_candidate_to_close'] = master_df['station_id'].isin(low_utilization_stations)

    # 7. 估算初始总容量并计算优化的容量上限
    initial_total_capacity = master_df['predicted_demand'].sum()
    capacity_upper_bound = 1.1 * initial_total_capacity
    print(f"估算初始总容量: {initial_total_capacity:,.0f}")
    print(f"优化模型的总容量上限 (C_max): {capacity_upper_bound:,.0f}")

    print("--- 数据预处理完成 ---\n")
    return master_df.set_index('station_id'), capacity_upper_bound


# =============================================================================
# 第二部分: 优化建模与求解
# =============================================================================

def solve_optimization_model(base_path, master_df, C_max, weight_cost, result_filename):
    """
    构建并求解共享单车站点优化的混合整数二次规划（MIQP）模型。

    参数:
        base_path (str): 保存结果文件的基础目录。
        master_df (pandas.DataFrame): 预处理后的主数据。
        C_max (float): 系统总容量的上限。
        weight_cost (float): 成本目标(F1)的权重 (0到1之间)。
        result_filename (str): 输出的Excel文件名。
    """
    print(f"--- 开始构建优化模型 ---")
    print(f"当前情景: 成本权重 = {weight_cost}, 服务权重 = {1 - weight_cost}")

    # 1. 从数据框中提取参数
    stations = master_df.index.tolist()
    D = master_df['predicted_demand'].to_dict()
    S = master_df['scheduling_cost_coeff'].to_dict()
    is_candidate = master_df['is_candidate_to_close'].to_dict()
    
    num_stations = len(stations)
    max_close_count = int(0.2 * num_stations)
    # 用于关联容量与站点状态的大M值
    M = master_df['predicted_demand'].max() * 1.5

    # 2. 创建COPT环境和模型
    env = cp.Envr()
    model = env.createModel(f"bike_opt_w{weight_cost}")

    # --- GPU 和求解器参数配置 ---
    print("正在配置 COPT 求解器参数以启用 GPU 加速...")
    try:
        # 将节点上的 LP/QP 求解算法设置为内点法（Barrier）。这是使用GPU的前提。
        # 参数值 2 对应于内点法。
        model.setParam(COPT.Param.LpMethod, 2)
        
        # 启用标准模式的GPU。模式2更激进，但会消耗更多显存。
        model.setParam(COPT.Param.GPUMode, 1)
        
        # 自动检测要使用的GPU设备（-1是默认值）。
        model.setParam(COPT.Param.GPUDevice, -1)

        # 验证设置
        print(f"  -> LpMethod 已设置为: {model.getParam(COPT.Param.LpMethod)} (2=内点法)")
        print(f"  -> GPUMode 已设置为: {model.getParam(COPT.Param.GPUMode)} (1=标准模式)")
        print("求解器已配置为使用 GPU 加速的内点法。")

    except Exception as e:
        print(f"警告: 无法设置GPU参数。求解器将仅使用CPU运行。错误: {e}", file=sys.stderr)
    
    model.setParam(COPT.Param.TimeLimit, 1800) # 设置30分钟的时间限制
    # --- 参数配置结束 ---

    # 3. 添加决策变量
    x = model.addVars(stations, vtype=COPT.BINARY, nameprefix="x") # x[i] = 1 如果站点 i 开放
    c = model.addVars(stations, vtype=COPT.INTEGER, nameprefix="c") # c[i] = 站点 i 的容量
    service_loss_vars = model.addVars(stations, vtype=COPT.CONTINUOUS, lb=0, nameprefix="u") # u[i] = 站点 i 的服务损失

    # 4. 添加模型约束
    print("正在构建模型约束...")
    model.addConstr(cp.quicksum(c[i] for i in stations) <= C_max, name="total_capacity")
    
    candidate_stations = [i for i in stations if is_candidate[i]]
    model.addConstr(cp.quicksum(1 - x[i] for i in candidate_stations) <= max_close_count, name="max_close_from_candidates")

    # 不在候选列表中的站点必须保持开放
    fixed_stations = [i for i in stations if not is_candidate[i]]
    model.addConstrs((x[i] == 1 for i in fixed_stations), nameprefix="fix_open_stations")

    # 将容量 'c' 与站点状态 'x' 关联起来
    model.addConstrs((c[i] <= M * x[i] for i in stations), nameprefix="cap_link_upper")
    model.addConstrs((c[i] >= x[i] for i in stations), nameprefix="cap_link_lower") # 如果开放，最小容量为1
    
    # 线性化服务损失项: u[i] >= D[i] - c[i]
    model.addConstrs((service_loss_vars[i] >= D[i] - c[i] for i in stations), nameprefix="service_loss_linearize")

    # 5. 定义两个目标函数
    F1_cost = cp.quicksum(S[i] * x[i] for i in stations) + cp.quicksum((D[i] - c[i])**2 for i in stations)
    F2_service_loss = cp.quicksum(service_loss_vars[i] for i in stations)

    # 6. 为加权和方法归一化目标函数
    print("正在计算目标函数的归一化范围...")
    # 为了找到理想点（最小值），分别为每个目标求解
    
    # --- BUG 修复 ---
    # 暂时关闭求解日志，以保持控制台输出整洁
    model.setParam(COPT.Param.Logging, 0)

    model.setObjective(F1_cost, COPT.MINIMIZE)
    model.solve()
    F1_min = model.objval if model.status in [COPT.OPTIMAL, COPT.FEASIBLE] else 0

    model.setObjective(F2_service_loss, COPT.MINIMIZE)
    model.solve()
    F2_min = model.objval if model.status in [COPT.OPTIMAL, COPT.FEASIBLE] else 0
    
    # 为最终求解恢复日志输出
    model.setParam(COPT.Param.Logging, 1)
    # --- 修复结束 ---

    # 估算天底点（最大值）以进行稳定的归一化
    # F1_max: 假设所有站点都开放(x=1)但容量最小(c=1)
    F1_max_est = sum(S.values()) + sum((D[i] - 1)**2 for i in stations)
    # F2_max: 假设最大的服务损失，即总需求（如果所有容量都为0）
    F2_max_est = sum(D.values())

    print(f"  -> 目标 F1 (成本) 范围: [{F1_min:,.2f}, {F1_max_est:,.2f}]")
    print(f"  -> 目标 F2 (服务) 范围: [{F2_min:,.2f}, {F2_max_est:,.2f}]")

    # 7. 设置最终的加权和目标函数
    F1_range = F1_max_est - F1_min
    F2_range = F2_max_est - F2_min

    if F1_range > 1e-6 and F2_range > 1e-6:
        print("正在设置归一化的加权和目标函数。")
        norm_F1 = (F1_cost - F1_min) / F1_range
        norm_F2 = (F2_service_loss - F2_min) / F2_range
        model.setObjective(weight_cost * norm_F1 + (1 - weight_cost) * norm_F2, COPT.MINIMIZE)
    else:
        print("警告: 目标函数范围为零，将使用原始目标函数。")
        model.setObjective(weight_cost * F1_cost + (1 - weight_cost) * F2_service_loss, COPT.MINIMIZE)

    # 8. 求解最终模型
    print("\n正在求解最终的 MIQP 模型...")
    model.solve()

    # 9. 分析并保存结果
    if model.status == COPT.OPTIMAL or model.status == COPT.FEASIBLE:
        print("\n求解成功，找到可行解！")
        print(f"求解器状态: {model.status_str}, 目标函数值: {model.objval:.4f}")
        
        solution = []
        closed_stations_count = 0
        total_final_capacity = 0
        for i in stations:
            is_open = int(round(x[i].x))
            capacity = int(round(c[i].x)) if is_open else 0
            if not is_open:
                closed_stations_count += 1
            total_final_capacity += capacity
            solution.append({
                'station_id': i,
                'is_retained': is_open,
                'capacity': capacity
            })
        
        result_df = pd.DataFrame(solution)
        
        print("\n--- 求解结果摘要 ---")
        print(f"保留站点总数: {num_stations - closed_stations_count} / {num_stations}")
        print(f"关闭站点总数: {closed_stations_count}")
        print(f"最终总容量: {total_final_capacity:,.0f} (上限为 {C_max:,.0f})")
        
        output_path = os.path.join(base_path, result_filename)
        result_df.to_excel(output_path, index=False)
        print(f"\n结果已成功保存至: {output_path}")
        print("结果文件前5行预览:")
        print(result_df.head())
    else:
        print(f"\n优化失败。求解器状态: {model.status_str} ({model.status})")
        print("将不会生成结果文件。")
    
    print(f"--- 权重为 {weight_cost} 的情景优化结束 ---\n")


# =============================================================================
# 主程序执行入口
# =============================================================================
if __name__ == '__main__':
    try:
        # 重要提示: 请将此路径设置为包含您的 CSV 文件的目录。
        base_folder_path = r"C:\Users\hk\fire_flower\attachment"
        
        # 第一步: 运行一次数据预处理
        master_data, C_max_val = preprocess_data(base_folder_path)

        # 第二步: 针对不同情景运行优化
        
        # 情景 1: 成本优先 (成本权重 = 0.8)
        solve_optimization_model(
            base_path=base_folder_path,
            master_df=master_data.copy(), # 使用数据的副本以避免修改原始数据
            C_max=C_max_val,
            weight_cost=0.8,
            result_filename="result1_1_cost_priority_cn.xlsx"
        )

        # 情景 2: 服务优先 (成本权重 = 0.2)
        solve_optimization_model(
            base_path=base_folder_path,
            master_df=master_data.copy(),
            C_max=C_max_val,
            weight_cost=0.2,
            result_filename="result1_2_service_priority_cn.xlsx"
        )

    except FileNotFoundError as e:
        print(f"\n严重错误: 找不到所需的数据文件。请仔细检查 'base_folder_path' 的路径设置。")
        print(f"详细信息: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\n程序执行期间发生意外错误: {e}", file=sys.stderr)
        # 为了调试，您可能希望打印完整的错误追溯信息
        import traceback
        traceback.print_exc()

