# 问题一：站点布局与容量配置优化 文件清单与说明

此目录 `fire_flower/qwencli/qs1` 包含了完成“问题一：站点布局与容量配置优化”任务的所有相关文件。

## 1. 核心文档

-   `station_optimization_model_design.md`: 站点优化模型的详细设计文档，包括问题分析、数学模型构建思路和实现建议。
-   `station_optimization_report.md`: 本次站点优化任务的综合分析报告，详细解释了模型、方法、结果和结论。

## 2. 核心代码文件

-   `generate_station_data.py`: 用于生成模拟站点和需求数据的Python脚本。
-   `station_optimization_model_linear.py`: 基于线性化目标函数（忽略二次项）的优化模型实现脚本。
-   `station_optimization_model_quadratic.py`: 对比研究脚本，实现了两种方法：忽略二次项和线性化二次项。

## 3. 数据文件

-   `station_data_input.xlsx`: 由 `generate_station_data.py` 生成的模拟输入数据，包含100个站点的ID和预测净需求量。

## 4. 优化结果文件

根据不同的优化模型，生成了两组结果：

### 方法一：忽略 $(d_i - c_i)^2$ 项

-   `result1_1_method1.xlsx`: 站点的保留/关闭状态。
-   `result1_2_method1.xlsx`: 保留站点的容量配置。

### 方法二：线性化 $(d_i - c_i)^2$ 项 (近似为 $|d_i - c_i|$)

-   `result1_1_method2.xlsx`: 站点的保留/关闭状态。
-   `result1_2_method2.xlsx`: 保留站点的容量配置。

## 5. 综合结果文件

-   `result1_1.xlsx` 和 `result1_2.xlsx`: 最初生成的结果文件（对应方法一）。

所有与问题一相关的文件均已生成并存放在此目录中。