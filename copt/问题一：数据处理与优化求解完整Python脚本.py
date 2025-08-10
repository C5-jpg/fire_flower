# bike_optimization_solution_fixed.py
# 修复版：处理 COPT 参数/求解/状态判断 等问题
# 2025-08-10

import os
import sys
import pandas as pd
import numpy as np
import coptpy as cp
from coptpy import COPT

def preprocess_data(base_path):
    print("--- 开始数据预处理 ---")
    trip_data_path = os.path.join(base_path, "202503-capitalbikeshare-tripdata.csv")
    demand_features_path = os.path.join(base_path, "demand_features.csv")

    if not os.path.exists(trip_data_path) or not os.path.exists(demand_features_path):
        raise FileNotFoundError(f"请检查数据文件是否存在: {trip_data_path} , {demand_features_path}")

    df_trip = pd.read_csv(trip_data_path)
    df_demand = pd.read_csv(demand_features_path)

    # 如果 df_demand 已经包含 borrow_count/return_count，则丢弃，重新计算
    for col in ['borrow_count', 'return_count']:
        if col in df_demand.columns:
            df_demand = df_demand.drop(columns=[col])

    df_trip.dropna(subset=['start_station_id', 'end_station_id'], inplace=True)
    df_trip['start_station_id'] = df_trip['start_station_id'].astype(int)
    df_trip['end_station_id'] = df_trip['end_station_id'].astype(int)
    df_demand.dropna(subset=['station_id'], inplace=True)
    df_demand['station_id'] = df_demand['station_id'].astype(int)

    borrow_counts = df_trip['start_station_id'].value_counts().reset_index()
    borrow_counts.columns = ['station_id', 'borrow_count']
    return_counts = df_trip['end_station_id'].value_counts().reset_index()
    return_counts.columns = ['station_id', 'return_count']

    master_df = pd.merge(df_demand, borrow_counts, on='station_id', how='left')
    master_df = pd.merge(master_df, return_counts, on='station_id', how='left')
    master_df[['borrow_count', 'return_count']] = master_df[['borrow_count', 'return_count']].fillna(0)

    master_df['borrow_return_diff'] = master_df['borrow_count'] - master_df['return_count']

    def sum_of_divisors(n):
        n = abs(int(n))
        if n == 0:
            return 0
        s = 0
        for i in range(1, int(np.sqrt(n)) + 1):
            if n % i == 0:
                s += i
                if i * i != n:
                    s += n // i
        return s

    master_df['scheduling_cost_coeff'] = master_df['borrow_return_diff'].apply(sum_of_divisors)
    master_df['turnover'] = master_df['borrow_count'] + master_df['return_count']

    n_stations = len(master_df)
    n_to_close_candidates = int(0.2 * n_stations)
    low_util = master_df.nsmallest(n_to_close_candidates, 'turnover')['station_id']
    master_df['is_candidate_to_close'] = master_df['station_id'].isin(low_util)

    initial_total_capacity = master_df['predicted_demand'].sum()
    capacity_upper_bound = 1.1 * initial_total_capacity

    print(f"估算初始总容量: {initial_total_capacity:,.0f}")
    print(f"优化后的容量上限 C_max: {capacity_upper_bound:,.0f}")
    print("--- 数据预处理完成 ---\n")

    return master_df.set_index('station_id'), capacity_upper_bound

def model_has_solution(model):
    """
    稳健地判断模型是否存在可行解：
      - 如果 model.status == COPT.OPTIMAL -> True
      - 否则查询属性 HasMipSol / HasLpSol
      - 最后做一个 objval 的兜底检查
    """
    try:
        if getattr(model, "status", None) == COPT.OPTIMAL:
            return True
        # 使用属性接口判断是否找到过解（整数/连续）
        try:
            has_mip = bool(model.getAttr("HasMipSol"))
        except Exception:
            has_mip = False
        try:
            has_lp = bool(model.getAttr("HasLpSol"))
        except Exception:
            has_lp = False
        if has_mip or has_lp:
            return True
        # 兜底：看 objval 是否存在（注意：objval 可能为 0）
        return getattr(model, "objval", None) is not None
    except Exception:
        return False

def solve_optimization_model(base_path, master_df, C_max, weight_cost, result_filename):
    print(f"\n--- 开始求解优化模型 (weight_cost={weight_cost}) ---")
    stations = master_df.index.tolist()
    D = master_df['predicted_demand'].to_dict()
    S = master_df['scheduling_cost_coeff'].to_dict()
    is_candidate = master_df['is_candidate_to_close'].to_dict()

    num_stations = len(stations)
    max_close_count = int(0.2 * num_stations)
    M = float(master_df['predicted_demand'].max() * 1.5)

    env = cp.Envr()
    model = env.createModel(f"bike_opt_w{weight_cost}")

    # --- 参数（GPU/算法/日志）设置（稳健写法） ---
    try:
        # LpMethod=2 表示 Barrier/内点法（部分版本可能用 2 表示内点）
        model.setParam(COPT.Param.LpMethod, 2)
        # 启用 GPU 模式（如果没有 GPU 或许可，COPT 会报错或返回失败）
        model.setParam(COPT.Param.GPUMode, 1)
        model.setParam(COPT.Param.GPUDevice, -1)  # -1 表示自动选择
        print("尝试启用 GPU 模式 (GPUMode=1)。如果没有 GPU/许可，会在下面捕获异常并继续使用 CPU。")
    except Exception as e:
        print(f"警告: 无法设置 GPU 参数或内点法，回退至默认求解器设置。详细: {e}", file=sys.stderr)

    # 控制求解日志：中间求解静默，最终求解可打开或继续静默
    model.setParam(COPT.Param.Logging, 0)
    model.setParam(COPT.Param.TimeLimit, 1800)  # 30 分钟时限（按需调整）

    # --- 变量 ---
    # 给 c 加上上界 ub=M，避免未界定的上界
    x = model.addVars(stations, vtype=COPT.BINARY, nameprefix="x")
    c = model.addVars(stations, lb=0.0, ub=M, vtype=COPT.INTEGER, nameprefix="c")
    u = model.addVars(stations, vtype=COPT.CONTINUOUS, lb=0.0, nameprefix="u")  # 服务损失的线性化变量

    # --- 约束 ---
    model.addConstr(cp.quicksum(c[i] for i in stations) <= C_max, name="total_capacity")

    candidate_stations = [i for i in stations if is_candidate[i]]
    model.addConstr(cp.quicksum(1 - x[i] for i in candidate_stations) <= max_close_count,
                    name="max_close_from_candidates")

    fixed_stations = [i for i in stations if not is_candidate[i]]
    if fixed_stations:
        model.addConstrs((x[i] == 1 for i in fixed_stations), nameprefix="force_open")

    model.addConstrs((c[i] <= M * x[i] for i in stations), nameprefix="cap_link_upper")
    model.addConstrs((c[i] >= x[i] for i in stations), nameprefix="cap_link_lower")
    model.addConstrs((u[i] >= D[i] - c[i] for i in stations), nameprefix="service_loss_linearize")
    model.addConstrs((u[i] >= 0 for i in stations), nameprefix="service_loss_nonneg")

    # --- 目标 ---
    # F1: 调度成本 + 空置平方成本 (二次项)
    F1_cost = cp.quicksum(S[i] * x[i] for i in stations) + cp.quicksum((D[i] - c[i]) * (D[i] - c[i]) for i in stations)
    F2_service_loss = cp.quicksum(u[i] for i in stations)

    # --- 归一化：先分别求得两个目标的最小值（理想点） ---
    print("正在求解以获取目标函数最小值（用于归一化），中间求解静默进行...")

    # 求 F1_min
    model.setObjective(F1_cost, COPT.MINIMIZE)
    try:
        model.solve()  # 无参数，使用 setParam 控制日志
    except Exception as e:
        print(f"[警告] 求解 F1_min 时出错: {e}", file=sys.stderr)

    F1_min = model.objval if model_has_solution(model) else 0.0
    print(f"  -> F1_min 估计: {F1_min:.4f} (状态: {getattr(model,'status_str', str(getattr(model,'status',None)))})")

    # 求 F2_min
    model.setObjective(F2_service_loss, COPT.MINIMIZE)
    try:
        model.solve()
    except Exception as e:
        print(f"[警告] 求解 F2_min 时出错: {e}", file=sys.stderr)

    F2_min = model.objval if model_has_solution(model) else 0.0
    print(f"  -> F2_min 估计: {F2_min:.4f} (状态: {getattr(model,'status_str', str(getattr(model,'status',None)))})")

    # 估算最大值用于归一化（保守估计，避免再次最大化求解）
    F1_max_est = sum(S.values()) + sum((D[i] - 1)**2 for i in stations)
    F2_max_est = sum(D.values())

    print(f"  -> 估计 F1 范围: [{F1_min:.4f}, {F1_max_est:.4f}]")
    print(f"  -> 估计 F2 范围: [{F2_min:.4f}, {F2_max_est:.4f}]")

    # --- 设置最终加权目标（归一化） ---
    model.setParam(COPT.Param.Logging, 1)  # 最终求解把日志打开（可改为 0）
    F1_range = F1_max_est - F1_min
    F2_range = F2_max_est - F2_min

    if F1_range > 1e-8 and F2_range > 1e-8:
        norm_F1 = (F1_cost - F1_min) / F1_range
        norm_F2 = (F2_service_loss - F2_min) / F2_range
        model.setObjective(weight_cost * norm_F1 + (1.0 - weight_cost) * norm_F2, COPT.MINIMIZE)
        print("使用归一化的加权和目标函数进行最终求解。")
    else:
        print("警告: 归一化范围过小，使用原始加权目标进行最终求解。")
        model.setObjective(weight_cost * F1_cost + (1.0 - weight_cost) * F2_service_loss, COPT.MINIMIZE)

    # --- 最终求解 ---
    print("\n开始最终求解（请耐心等待求解器输出）...")
    try:
        model.solve()
    except Exception as e:
        print(f"求解过程出错: {e}", file=sys.stderr)

    # --- 解析结果 ---
    status_str = getattr(model, "status_str", None)
    print(f"Solver finished. status: {getattr(model,'status', model.status)}  status_str: {status_str}")

    if model_has_solution(model):
        print("找到可行解/最优解，开始解析变量值...")
        solution = []
        closed_cnt = 0
        total_cap = 0
        for i in stations:
            xi = int(round(x[i].x))
            ci = int(round(c[i].x)) if xi == 1 else 0
            if xi == 0:
                closed_cnt += 1
            total_cap += ci
            solution.append({'station_id': i, 'is_retained': xi, 'capacity': ci})

        result_df = pd.DataFrame(solution)
        out_path = os.path.join(base_path, result_filename)
        result_df.to_excel(out_path, index=False)
        print(f"结果保存到: {out_path}")
        print("结果前 5 行预览：")
        print(result_df.head())
        print(f"关闭站点数: {closed_cnt} / {num_stations}， 最终总容量: {total_cap:.0f} (上限 C_max={C_max:.0f})")
    else:
        print("未找到可行解（或求解器未返回有效解）。请检查模型/数据/参数设置。")

    print(f"--- 情景 weight_cost={weight_cost} 求解结束 ---\n")


if __name__ == "__main__":
    base_folder_path = r"C:\Users\hk\fire_flower\attachment"

    try:
        master_data, C_max_val = preprocess_data(base_folder_path)

        solve_optimization_model(base_folder_path, master_data.copy(), C_max_val, weight_cost=0.8,
                                 result_filename="result1_1_cost_priority_fixed.xlsx")

        solve_optimization_model(base_folder_path, master_data.copy(), C_max_val, weight_cost=0.2,
                                 result_filename="result1_2_service_priority_fixed.xlsx")

    except Exception as e:
        print("执行期间捕获到异常：", e, file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
