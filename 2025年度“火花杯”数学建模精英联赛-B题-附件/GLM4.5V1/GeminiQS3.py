import pandas as pd
import numpy as np
import coptpy as cp
from coptpy import COPT
import os

def solve_problem_3(base_path, master_df):
    """
    构建并求解问题三的综合成本效益模型 (MIQP)。

    参数:
    base_path (str): 存放所有附件的文件夹路径。
    master_df (pandas.DataFrame): 从问题一预处理好的主数据框。
    """
    
    # --- 0. 创建输出文件夹 ---
    # 根据题目要求，在qwencli文件夹下创建一个qs3文件夹
    output_dir = os.path.join(base_path, "qwencli", "qs3")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")

    result_filename = os.path.join(output_dir, "result3.xlsx")
    
    print("\n--- 开始求解问题三：综合成本效益模型 ---")
    
    # --- 1. 定义模型参数 ---
    
    # 定义用户体验损失惩罚系数 P
    # 理由：该值参考了本市乘坐出租车完成类似短途出行的平均溢价成本（约15元），
    # 体现了对用户时间价值和出行便利性的高度重视。
    P_penalty_coefficient = 15.0
    print(f"用户体验损失惩罚系数 (P) 设置为: {P_penalty_coefficient}")

    stations = master_df.index.tolist()
    D = master_df['predicted_demand'].to_dict()
    S = master_df['scheduling_cost_coeff'].to_dict()
    is_candidate = master_df['is_candidate_to_close'].to_dict()
    
    C_max = master_df.capacity_upper_bound
    max_close_count = int(0.2 * len(stations))
    M = master_df['predicted_demand'].max() * 1.5 # Big-M

    # --- 2. 创建COPT环境和模型 ---
    env = cp.Envr()
    model = env.createModel("bike_optimization_p3")

    # --- 3. 添加决策变量 ---
    x = model.addVars(stations, vtype=COPT.BINARY, nameprefix="x")       # 站点是否保留
    c = model.addVars(stations, vtype=COPT.INTEGER, nameprefix="c")     # 站点容量
    u = model.addVars(stations, vtype=COPT.CONTINUOUS, nameprefix="u")  # 用户体验损失 (服务缺口)

    # --- 4. 添加约束 (与问题一/二相同) ---
    model.addConstr(cp.quicksum(c[i] for i in stations) <= C_max, name="total_capacity")
    
    candidate_stations = [i for i in stations if is_candidate[i]]
    model.addConstr(cp.quicksum(1 - x[i] for i in candidate_stations) <= max_close_count, name="max_close")

    fixed_stations = [i for i in stations if not is_candidate[i]]
    model.addConstrs((x[i] == 1 for i in fixed_stations), name="fixed_stations")

    model.addConstrs((c[i] <= M * x[i] for i in stations), nameprefix="cap_link_upper")
    model.addConstrs((c[i] >= x[i] for i in stations), nameprefix="cap_link_lower")
    
    # 用户体验损失线性化约束
    model.addConstrs((u[i] >= D[i] - c[i] for i in stations), nameprefix="user_loss_linearize")
    model.addConstrs((u[i] >= 0 for i in stations), nameprefix="user_loss_non_negative")

    # --- 5. 定义并设置综合成本目标函数 ---
    cost_dispatch = cp.quicksum(S[i] * x[i] for i in stations)
    cost_vacancy = cp.quicksum((D[i] - c[i]) * (D[i] - c[i]) for i in stations)
    cost_user_exp = cp.quicksum(P_penalty_coefficient * u[i] for i in stations)
    
    total_social_cost = cost_dispatch + cost_vacancy + cost_user_exp
    
    model.setObjective(total_social_cost, COPT.MINIMIZE)
    
    print("模型构建完成，开始使用COPT求解...")
    # 可以设置求解器参数，例如时间限制
    # model.setParam(COPT.Param.TimeLimit, 300) # 限制求解时间为300秒

    # --- 6. 求解模型 ---
    model.solve()

    # --- 7. 分析和保存结果 ---
    if model.status == COPT.OPTIMAL or model.status == COPT.FEASIBLE:
        if model.status == COPT.OPTIMAL:
            print("找到最优解!")
        else:
            print("在规定时间内找到一个可行解。")
            
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
# 主程序入口 (示例)
# =============================================================================
# 为了运行此脚本，你需要先运行问题一的代码来生成 master_data
# 这里我们假设 master_data 已经存在
if __name__ == '__main__':
    # 假设这是您存放附件的文件夹路径
    base_folder_path = "C:/Users/hk/fire_flower/2025年度“火花杯”数学建模精英联赛-B题-附件"
    
    # 这是一个示例，实际使用时你需要先通过问题一的代码获得 master_df
    # from GeminiQS1 import preprocess_data # 假设你的问题一代码文件名为 GeminiQS1.py
    # master_data = preprocess_data(base_folder_path)

    # --- 为了让此脚本能独立演示，我们创建一个模拟的 master_df ---
    print("警告: 正在使用模拟数据进行演示。请确保您已通过问题一的代码生成真实的 master_data。")
    sample_station_ids = range(31000, 31100)
    mock_data = {
        'predicted_demand': np.random.randint(50, 500, size=100),
        'scheduling_cost_coeff': np.random.randint(100, 10000, size=100),
        'is_candidate_to_close': np.random.choice([True, False], size=100, p=[0.2, 0.8]),
    }
    master_data_mock = pd.DataFrame(mock_data, index=sample_station_ids)
    master_data_mock.index.name = 'station_id'
    master_data_mock.capacity_upper_bound = master_data_mock['predicted_demand'].sum() * 1.1
    # --- 模拟数据结束 ---
    
    # 使用模拟数据运行问题三求解器
    solve_problem_3(base_folder_path, master_data_mock)

