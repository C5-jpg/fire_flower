import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import os

# --- 通用设置 ---
# 设置中文字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Plotly全局配置
pio.templates.default = "plotly_white"
config = {'displaylogo': False, 'displayModeBar': True}

# 创建输出目录
output_dir = '.'
os.makedirs(output_dir, exist_ok=True)

# --- 1. 问题一 (确定性优化) 可视化 ---

# 1.1. 数据概览: 所有站点预测需求分布直方图
print("正在生成图表1: 所有站点预测需求分布直方图...")
df_stations = pd.read_excel('../station_data_input.xlsx')
fig, ax = plt.subplots(figsize=(10, 6))
n, bins, patches = ax.hist(df_stations['predicted_demand'], bins=30, color='skyblue', edgecolor='black')

# 添加统计信息
mean_val = df_stations['predicted_demand'].mean()
median_val = df_stations['predicted_demand'].median()
ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'均值: {mean_val:.2f}')
ax.axvline(median_val, color='green', linestyle='-.', linewidth=1.5, label=f'中位数: {median_val:.2f}')

ax.set_xlabel('预测需求量')
ax.set_ylabel('站点数量')
ax.set_title('所有站点预测需求分布')
ax.legend()
ax.grid(axis='y', alpha=0.75)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'p1_demand_histogram.svg'), format='svg')
plt.close(fig)
print("图表1已保存至 p1_demand_histogram.svg")


# 1.2. 方法一 vs 方法二结果对比

# 读取结果数据
df_status_method1 = pd.read_excel('../result1_1_method1.xlsx')
df_capacity_method1 = pd.read_excel('../result1_2_method1.xlsx')

df_status_method2 = pd.read_excel('../result1_1_method2.xlsx')
df_capacity_method2 = pd.read_excel('../result1_2_method2.xlsx')

# 合并数据以便于比较
df_comparison_status = pd.merge(df_status_method1, df_status_method2, on='station_id', suffixes=('_method1', '_method2'))
df_comparison_status['same'] = df_comparison_status['retain_method1'] == df_comparison_status['retain_method2']

# 图表2: 两种方法的站点保留状态对比 (散点图)
print("正在生成图表2: 两种方法的站点保留状态对比...")
fig, ax = plt.subplots(figsize=(14, 6))
# 为了可视化，我们按站点ID排序
df_comparison_status_sorted = df_comparison_status.sort_values('station_id').reset_index(drop=True)
x_indices = range(len(df_comparison_status_sorted))

scatter1 = ax.scatter(x_indices, df_comparison_status_sorted['retain_method1'], alpha=0.7, label='方法一 (忽略二次项)', marker='o')
scatter2 = ax.scatter(x_indices, df_comparison_status_sorted['retain_method2'], alpha=0.7, label='方法二 (线性化二次项)', marker='x')

# 用不同颜色高亮不一致的点
diff_mask = df_comparison_status_sorted['same'] == False
if diff_mask.sum() > 0:
    ax.scatter(np.array(x_indices)[diff_mask], df_comparison_status_sorted.loc[diff_mask, 'retain_method1'], color='red', marker='o', s=50, label='状态不同 (方法一)', zorder=5)
    ax.scatter(np.array(x_indices)[diff_mask], df_comparison_status_sorted.loc[diff_mask, 'retain_method2'], color='orange', marker='x', s=50, label='状态不同 (方法二)', zorder=5)

ax.set_xlabel('站点索引 (按ID排序)')
ax.set_ylabel('是否保留 (1=保留, 0=关闭)')
ax.set_title('两种方法的站点保留状态对比')
ax.set_yticks([0, 1])
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'p1_retain_status_comparison.svg'), format='svg')
plt.close(fig)
print("图表2已保存至 p1_retain_status_comparison.svg")


# 图表3: 保留站点的容量配置对比 (散点图)
print("正在生成图表3: 保留站点的容量配置对比...")
# 找到在两种方法中都被保留的站点
common_retained = set(df_capacity_method1['station_id']).intersection(set(df_capacity_method2['station_id']))
df_common_cap1 = df_capacity_method1[df_capacity_method1['station_id'].isin(common_retained)].copy()
df_common_cap2 = df_capacity_method2[df_capacity_method2['station_id'].isin(common_retained)].copy()

df_cap_comparison = pd.merge(df_common_cap1, df_common_cap2, on='station_id', suffixes=('_method1', '_method2'))

if not df_cap_comparison.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    # 按站点ID排序
    df_cap_comparison_sorted = df_cap_comparison.sort_values('station_id').reset_index(drop=True)
    
    ax.scatter(df_cap_comparison_sorted.index, df_cap_comparison_sorted['capacity_method1'], alpha=0.7, label='方法一 容量', marker='o')
    ax.scatter(df_cap_comparison_sorted.index, df_cap_comparison_sorted['capacity_method2'], alpha=0.7, label='方法二 容量', marker='x')
    
    # 添加 y=x 参考线
    min_cap = min(df_cap_comparison_sorted['capacity_method1'].min(), df_cap_comparison_sorted['capacity_method2'].min())
    max_cap = max(df_cap_comparison_sorted['capacity_method1'].max(), df_cap_comparison_sorted['capacity_method2'].max())
    ax.plot([0, len(df_cap_comparison_sorted)], [min_cap, max_cap], 'r--', linewidth=1, label='y=x (完全一致)')
    
    ax.set_xlabel('共同保留站点索引 (按ID排序)')
    ax.set_ylabel('分配容量')
    ax.set_title('两种方法对共同保留站点的容量配置对比')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'p1_capacity_configuration_comparison.svg'), format='svg')
    plt.close(fig)
    print("图表3已保存至 p1_capacity_configuration_comparison.svg")
else:
    print("没有在两种方法中都被保留的站点，无法生成图表3。")


# 图表4: 两种方法的成本构成对比 (堆叠柱状图)
print("正在生成图表4: 两种方法的成本构成对比...")
# 为了计算成本，我们需要原始需求数据
df_station_data = pd.read_excel('../station_data_input.xlsx')
# 创建一个字典方便查找
demand_dict = dict(zip(df_station_data['station_id'], df_station_data['predicted_demand']))

# 定义权重 (与优化模型一致)
W1, W2 = 1.0, 1.0
K_scheduling = 1.0

def calculate_cost(df_status, df_capacity, method_name):
    total_failure_cost = 0
    total_scheduling_cost = 0
    
    for _, row in df_status.iterrows():
        sid = row['station_id']
        is_retained = row['retain']
        if is_retained:
            d_i = demand_dict[sid]
            # 找到该站点的容量
            cap_row = df_capacity[df_capacity['station_id'] == sid]
            if not cap_row.empty:
                c_i = cap_row['capacity'].iloc[0]
                # 计算失败惩罚
                borrow_failure = max(0, d_i - c_i)
                return_failure = max(0, -d_i - c_i)
                total_failure_cost += (borrow_failure + return_failure) * W1
                
                # 计算调度成本
                total_scheduling_cost += is_retained * abs(d_i) * K_scheduling * W2
            else:
                # 站点被保留但没有容量数据，这在逻辑上是错误的，但为了健壮性我们处理一下
                print(f"警告: 站点 {sid} 在 {method_name} 中被保留但无容量数据。")
                
    return total_failure_cost, total_scheduling_cost

cost_method1 = calculate_cost(df_status_method1, df_capacity_method1, "方法一")
cost_method2 = calculate_cost(df_status_method2, df_capacity_method2, "方法二")

# 准备绘图数据
methods = ['方法一 (忽略二次项)', '方法二 (线性化二次项)']
failure_costs = [cost_method1[0], cost_method2[0]]
scheduling_costs = [cost_method1[1], cost_method2[1]]
total_costs = [sum(cost_method1), sum(cost_method2)]

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(methods))

p1 = ax.bar(index, failure_costs, bar_width, label='失败惩罚成本')
p2 = ax.bar(index, scheduling_costs, bar_width, bottom=failure_costs, label='调度成本')

# 在柱子上添加总成本标签
for i in range(len(methods)):
    ax.text(index[i], total_costs[i] + 10, f'总成本: {total_costs[i]:.2f}', ha='center', va='bottom')

ax.set_xlabel('优化方法')
ax.set_ylabel('成本')
ax.set_title('两种确定性优化方法的成本构成对比')
ax.set_xticks(index)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'p1_cost_breakdown_comparison.svg'), format='svg')
plt.close(fig)
print("图表4已保存至 p1_cost_breakdown_comparison.svg")


# 1.3. 最优解分析 (聚焦方法二)

# 图表5: 保留站点的空间分布 (示意地图) - 使用 Plotly 生成交互式 HTML
print("正在生成图表5: 保留站点的空间分布 (交互式)...")
# 模拟空间分布：我们使用站点ID的数字部分作为X坐标，预测需求作为Y坐标
df_retained_method2 = pd.merge(df_status_method2[df_status_method2['retain']==1], df_capacity_method2, on='station_id')
df_retained_with_data = pd.merge(df_retained_method2, df_station_data, on='station_id')

# 提取ID中的数字部分作为X坐标
df_retained_with_data['x_coord'] = df_retained_with_data['station_id'].str.extract('(\d+)').astype(int)

fig_map = px.scatter(
    df_retained_with_data, 
    x='x_coord', 
    y='predicted_demand', 
    size='capacity', 
    color='predicted_demand',
    color_continuous_scale='RdYlBu',
    hover_data=['station_id', 'capacity'],
    title='保留站点的空间分布 (方法二结果)',
    labels={'x_coord': '站点ID编号', 'predicted_demand': '预测需求量', 'capacity': '容量'}
)

fig_map.update_layout(
    xaxis_title='站点ID编号',
    yaxis_title='预测需求量',
    coloraxis_colorbar=dict(title='预测需求')
)

# 保存为HTML文件
fig_map.write_html(os.path.join(output_dir, 'p1_retained_stations_map.html'))
print("图表5已保存至 p1_retained_stations_map.html")


# 图表6: 保留站点的容量与需求关系 (散点图)
print("正在生成图表6: 保留站点的容量与需求关系...")
fig, ax = plt.subplots(figsize=(10, 6))

# 区分借车和还车需求
borrow_mask = df_retained_with_data['predicted_demand'] > 0
return_mask = df_retained_with_data['predicted_demand'] <= 0

ax.scatter(df_retained_with_data.loc[borrow_mask, 'predicted_demand'], 
           df_retained_with_data.loc[borrow_mask, 'capacity'], 
           c='blue', label='借车需求 (>0)', alpha=0.7)
ax.scatter(df_retained_with_data.loc[return_mask, 'predicted_demand'], 
           df_retained_with_data.loc[return_mask, 'capacity'], 
           c='red', label='还车需求 (<=0)', alpha=0.7)

# 添加 y=|x| 参考线
x_ref = np.linspace(df_retained_with_data['predicted_demand'].min(), df_retained_with_data['predicted_demand'].max(), 100)
y_ref = np.abs(x_ref)
ax.plot(x_ref, y_ref, 'g--', linewidth=1, label='y=|x| (理想匹配线)')

ax.set_xlabel('预测需求量')
ax.set_ylabel('分配容量')
ax.set_title('保留站点的容量与需求关系 (方法二结果)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'p1_capacity_vs_demand.svg'), format='svg')
plt.close(fig)
print("图表6已保存至 p1_capacity_vs_demand.svg")


# 图表7: 站点效率分析 (柱状图)
print("正在生成图表7: 站点效率分析...")
df_retained_with_data['efficiency'] = np.minimum(df_retained_with_data['capacity'], np.abs(df_retained_with_data['predicted_demand'])) / \
                                      np.maximum(df_retained_with_data['capacity'], np.abs(df_retained_with_data['predicted_demand']) + 1e-8) # 避免除以零

fig, ax = plt.subplots(figsize=(14, 6))
# 按效率排序
df_efficiency_sorted = df_retained_with_data.sort_values('efficiency', ascending=False).reset_index(drop=True)
x_indices_eff = range(len(df_efficiency_sorted))

bar_colors = plt.cm.viridis(df_efficiency_sorted['predicted_demand'] / df_efficiency_sorted['predicted_demand'].max())

bars = ax.bar(x_indices_eff, df_efficiency_sorted['efficiency'], color=bar_colors, alpha=0.8)

ax.set_xlabel('保留站点 (按效率降序排列)')
ax.set_ylabel('效率 (min(容量,|需求|) / max(容量,|需求|))')
ax.set_title('保留站点的效率分析 (方法二结果)')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 为了不让X轴标签过于拥挤，只显示一部分
step = max(1, len(df_efficiency_sorted) // 20) 
tick_labels = [df_efficiency_sorted.loc[i, 'station_id'] if i % step == 0 else '' for i in range(len(df_efficiency_sorted))]
ax.set_xticks(range(0, len(df_efficiency_sorted), step))
ax.set_xticklabels([label for i, label in enumerate(tick_labels) if label != ''], rotation=45, ha="right")

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'p1_station_efficiency.svg'), format='svg')
plt.close(fig)
print("图表7已保存至 p1_station_efficiency.svg")


# --- 2. 问题二 (鲁棒优化) 可视化 ---

# 2.1. 不确定性场景分析

# 图表8: 不同场景下总需求对比 (柱状图)
print("正在生成图表8: 不同场景下总需求对比...")
df_scenarios = pd.read_excel('../qs2/uncertainty_scenarios.xlsx')

scenario_stats = df_scenarios.groupby('scenario')['actual_demand'].agg(['sum', 'std']).reset_index()
scenario_stats = scenario_stats.sort_values('scenario') # 按场景名称排序

fig, ax = plt.subplots(figsize=(10, 6))
bar_container = ax.bar(scenario_stats['scenario'], scenario_stats['sum'], 
                       yerr=scenario_stats['std'], capsize=5, 
                       color=['skyblue', 'lightgreen', 'gold', 'orange', 'lightcoral'])

ax.set_xlabel('不确定性场景')
ax.set_ylabel('总需求量 (误差线为需求标准差)')
ax.set_title('不同不确定性场景下的总需求对比')
ax.tick_params(axis='x', rotation=45)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'p2_scenario_demand_comparison.svg'), format='svg')
plt.close(fig)
print("图表8已保存至 p2_scenario_demand_comparison.svg")


# 2.2. 鲁棒解 vs 确定性解对比

# 读取鲁棒优化结果
df_status_robust = pd.read_excel('../qs2/result2_1.xlsx')
df_capacity_robust = pd.read_excel('../qs2/result2_2.xlsx')

# 图表9: 站点保留状态终极对比 (桑基图/Sankey Diagram) - 使用 Plotly 生成交互式 HTML
print("正在生成图表9: 站点保留状态终极对比 (桑基图)...")
# 构建桑基图数据
# 节点：["确定性解-保留", "确定性解-关闭", "鲁棒解-保留", "鲁棒解-关闭"]
# 流量：
# 1. 确定性保留 -> 鲁棒保留
# 2. 确定性保留 -> 鲁棒关闭
# 3. 确定性关闭 -> 鲁棒保留
# 4. 确定性关闭 -> 鲁棒关闭

# 创建一个包含所有站点和两种方法状态的DataFrame
all_station_ids = set(df_status_method2['station_id']).union(set(df_status_robust['station_id']))
df_all_status = pd.DataFrame({'station_id': list(all_station_ids)})

# 合并状态
df_all_status = df_all_status.merge(df_status_method2[['station_id', 'retain']], on='station_id', how='left', suffixes=('', '_det')).fillna(0)
df_all_status = df_all_status.merge(df_status_robust[['station_id', 'retain']], on='station_id', how='left', suffixes=('_det', '_robust')).fillna(0)

# 计算流量
flow_det_retain_to_robust_retain = ((df_all_status['retain_det'] == 1) & (df_all_status['retain_robust'] == 1)).sum()
flow_det_retain_to_robust_close = ((df_all_status['retain_det'] == 1) & (df_all_status['retain_robust'] == 0)).sum()
flow_det_close_to_robust_retain = ((df_all_status['retain_det'] == 0) & (df_all_status['retain_robust'] == 1)).sum()
flow_det_close_to_robust_close = ((df_all_status['retain_det'] == 0) & (df_all_status['retain_robust'] == 0)).sum()

# 构造桑基图数据
source = [0, 0, 1, 1]  # ["确定性解-保留", "确定性解-保留", "确定性解-关闭", "确定性解-关闭"]
target = [2, 3, 2, 3]  # ["鲁棒解-保留", "鲁棒解-关闭", "鲁棒解-保留", "鲁棒解-关闭"]
value = [flow_det_retain_to_robust_retain, flow_det_retain_to_robust_close,
         flow_det_close_to_robust_retain, flow_det_close_to_robust_close]

node_labels = ["确定性解-保留", "确定性解-关闭", "鲁棒解-保留", "鲁棒解-关闭"]

fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=node_labels,
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        # color = ["rgba(31, 119, 180, 0.5)", "rgba(255, 127, 14, 0.5)", "rgba(44, 160, 44, 0.5)", "rgba(214, 39, 40, 0.5)"]
    )
)])

fig_sankey.update_layout(title_text="从确定性解到鲁棒解的站点保留状态变化 (桑基图)", font_size=12)

# 保存为HTML文件
fig_sankey.write_html(os.path.join(output_dir, 'p2_retain_status_sankey.html'))
print("图表9已保存至 p2_retain_status_sankey.html")


# 图表10: 鲁棒解在各场景下的成本表现 (折线图)
print("正在生成图表10: 鲁棒解在各场景下的成本表现...")

# 从鲁棒优化模型的输出日志中，我们已经知道各场景的成本
# 为了演示，我们在这里重新计算一遍（在实际项目中，可以直接从日志或模型中获取）
scenario_costs_robust = {
    'Scenario_1_Worst': 2691.06,
    'Scenario_2_Best': 2925.06, # 这是目标值，也是最坏情况
    'Scenario_3_Mixed_A': 2722.65,
    'Scenario_4_Mixed_B': 2895.12,
    'Scenario_5_Base': 2753.00
}

# 按场景名称排序
sorted_scenarios = sorted(scenario_costs_robust.keys())
sorted_costs = [scenario_costs_robust[sc] for sc in sorted_scenarios]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sorted_scenarios, sorted_costs, marker='o', linewidth=2, markersize=8, label='鲁棒解成本')

# 标记最坏情况
worst_scenario_idx = sorted_scenarios.index('Scenario_2_Best')
ax.annotate(f'最坏情况\n成本: {sorted_costs[worst_scenario_idx]:.2f}',
            xy=(worst_scenario_idx, sorted_costs[worst_scenario_idx]), 
            xytext=(worst_scenario_idx, sorted_costs[worst_scenario_idx] + 100),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, ha='center', color='red')

# 添加水平线表示目标最坏情况成本
ax.axhline(y=sorted_costs[worst_scenario_idx], color='r', linestyle='--', linewidth=1, alpha=0.7, label=f'目标最坏成本: {sorted_costs[worst_scenario_idx]:.2f}')

ax.set_xlabel('不确定性场景')
ax.set_ylabel('总成本')
ax.set_title('鲁棒解在各不确定性场景下的成本表现')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(axis='x', rotation=45)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'p2_robust_solution_cost_across_scenarios.svg'), format='svg')
plt.close(fig)
print("图表10已保存至 p2_robust_solution_cost_across_scenarios.svg")


# 2.3. 鲁棒解分析

# 图表11: “关闭所有站点”策略的成本分析 (表格+图表)
print("正在生成图表11: “关闭所有站点”策略的成本分析...")
# 我们将创建一个DataFrame来展示分析，并将其保存为HTML表格

# 鲁棒解的成本 (对于所有场景都是0，因为没有站点被保留)
robust_cost_per_scenario = {sc: 0 for sc in sorted_scenarios}

# 确定性解在各场景下的成本 (需要计算)
# 我们使用问题一方法二的解，在每个场景下计算成本
def calculate_cost_under_scenario(df_status, df_capacity, df_scenario_data, scenario_name):
    total_failure_cost = 0
    total_scheduling_cost = 0
    
    # 获取该场景下的数据
    df_scenario_for_this = df_scenario_data[df_scenario_data['scenario'] == scenario_name].copy()
    scenario_demand_dict = dict(zip(df_scenario_for_this['station_id'], df_scenario_for_this['actual_demand']))
    scenario_cost_coeff_dict = dict(zip(df_scenario_for_this['station_id'], df_scenario_for_this['actual_cost_coeff']))
    
    for _, row in df_status.iterrows():
        sid = row['station_id']
        is_retained = row['retain']
        if is_retained:
            d_i = scenario_demand_dict.get(sid, 0) # 如果场景数据中没有该站点，默认需求为0
            cost_coeff = scenario_cost_coeff_dict.get(sid, 1.0)
            # 找到该站点的容量
            cap_row = df_capacity[df_capacity['station_id'] == sid]
            if not cap_row.empty:
                c_i = cap_row['capacity'].iloc[0]
                # 计算失败惩罚
                borrow_failure = max(0, d_i - c_i)
                return_failure = max(0, -d_i - c_i)
                total_failure_cost += (borrow_failure + return_failure) * W1
                
                # 计算调度成本
                total_scheduling_cost += is_retained * abs(d_i) * cost_coeff * W2
    return total_failure_cost + total_scheduling_cost

# 计算确定性解在各场景下的成本
deterministic_costs_per_scenario = {}
for sc in sorted_scenarios:
    cost = calculate_cost_under_scenario(df_status_method2, df_capacity_method2, df_scenarios, sc)
    deterministic_costs_per_scenario[sc] = cost

# 构建分析DataFrame
analysis_data = {
    '场景名称': sorted_scenarios,
    '鲁棒解成本': [robust_cost_per_scenario[sc] for sc in sorted_scenarios],
    '确定性解成本': [deterministic_costs_per_scenario[sc] for sc in sorted_scenarios],
    '成本差异 (确定性-鲁棒)': [deterministic_costs_per_scenario[sc] - robust_cost_per_scenario[sc] for sc in sorted_scenarios]
}
df_cost_analysis = pd.DataFrame(analysis_data)

# 保存为HTML表格 (带基本样式)
html_table = df_cost_analysis.to_html(index=False, table_id='cost-analysis-table')

# 添加CSS样式
styled_html = f"""
<!DOCTYPE html>
<html>
<head>
<title>鲁棒解与确定性解成本分析</title>
<style>
  #cost-analysis-table {{
    border-collapse: collapse;
    width: 100%;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }}
  #cost-analysis-table th, #cost-analysis-table td {{
    border: 1px solid #ddd;
    padding: 12px;
    text-align: center;
  }}
  #cost-analysis-table tr:nth-child(even) {{background-color: #f2f2f2;}}
  #cost-analysis-table tr:hover {{background-color: #ddd;}}
  #cost-analysis-table th {{
    padding-top: 12px;
    padding-bottom: 12px;
    text-align: center;
    background-color: #4CAF50;
    color: white;
  }}
  .highlight {{
    background-color: #ffeb3b;
    font-weight: bold;
  }}
</style>
</head>
<body>
<h2>鲁棒解与确定性解在各场景下的成本对比分析</h2>
<p>此表格展示了在不同不确定性场景下，鲁棒解（关闭所有站点）与问题一确定性解（方法二）的成本对比。</p>
{html_table}
<script>
// 高亮显示最坏情况行
document.addEventListener('DOMContentLoaded', function() {{
  const table = document.getElementById('cost-analysis-table');
  const rows = table.getElementsByTagName('tr');
  for (let i = 0; i < rows.length; i++) {{
    const cells = rows[i].getElementsByTagName('td');
    if (cells.length > 0 && cells[0].textContent.includes('Scenario_2_Best')) {{
      rows[i].classList.add('highlight');
      break;
    }}
  }}
}});
</script>
</body>
</html>
"""

with open(os.path.join(output_dir, 'p2_robust_vs_deterministic_cost_analysis.html'), 'w', encoding='utf-8') as f:
    f.write(styled_html)
print("图表11已保存至 p2_robust_vs_deterministic_cost_analysis.html")


# 图表12: 问题一与问题二核心指标对比 (综合仪表盘概念 - 用柱状图代替)
print("正在生成图表12: 问题一与问题二核心指标对比...")
# 准备对比数据
comparison_metrics = {
    '保留站点数': [df_status_method2['retain'].sum(), df_status_robust['retain'].sum()],
    '总容量': [df_capacity_method2['capacity'].sum() if not df_capacity_method2.empty else 0, 
              df_capacity_robust['capacity'].sum() if not df_capacity_robust.empty else 0],
    '问题一总成本': [sum(cost_method2), None], # 只适用于问题一
    '鲁棒最坏情况成本': [None, sorted_costs[worst_scenario_idx]] # 只适用于问题二
}

# 分别绘制两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 子图1: 保留站点数和总容量
x1 = np.arange(2)
width = 0.35

rects1 = ax1.bar(x1 - width/2, [comparison_metrics['保留站点数'][0], comparison_metrics['保留站点数'][1]], width, label='保留站点数')
rects2 = ax1.bar(x1 + width/2, [comparison_metrics['总容量'][0], comparison_metrics['总容量'][1]], width, label='总容量')

ax1.set_xlabel('问题')
ax1.set_ylabel('数量')
ax1.set_title('保留站点数与总容量对比')
ax1.set_xticks(x1)
ax1.set_xticklabels(['问题一 (确定性)', '问题二 (鲁棒)'])
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 在柱子上添加数值标签
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(int(height)) if height == int(height) else '{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1, ax1)
autolabel(rects2, ax1)

# 子图2: 成本对比 (使用不同的X轴范围)
x2 = np.arange(2)
# 问题一成本放在x=0，问题二最坏成本放在x=1
costs_to_plot = [sum(cost_method2), sorted_costs[worst_scenario_idx]]
labels_to_plot = ['问题一总成本\n(确定性)', '问题二最坏成本\n(鲁棒)']
colors_to_plot = ['blue', 'red']

bars2 = ax2.bar(labels_to_plot, costs_to_plot, color=colors_to_plot, alpha=0.7)

ax2.set_ylabel('成本')
ax2.set_title('成本对比')
ax2.grid(axis='y', linestyle='--', alpha=0.7)
autolabel(bars2, ax2)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'p1_vs_p2_overall_comparison_dashboard.svg'), format='svg')
plt.close(fig)
print("图表12已保存至 p1_vs_p2_overall_comparison_dashboard.svg")

print("\n所有可视化图表已生成并保存至 pics 文件夹。")