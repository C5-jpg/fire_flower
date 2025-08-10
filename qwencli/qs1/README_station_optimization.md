# 站点优化项目文件清单与说明

此目录 `fire_flower/qwencli` 包含了完成“站点布局与容量配置优化”任务的所有相关文件。

## 1. 核心代码文件

-   `station_optimization_model_design.md`: 站点优化模型的详细设计文档，包括问题分析、数学模型构建思路和实现建议。
-   `generate_station_data.py`: 用于生成模拟站点和需求数据的Python脚本。
-   `station_optimization_model_linear.py`: 基于线性化目标函数（忽略二次项）的优化模型实现脚本。
-   `station_optimization_model_quadratic.py`: 对比研究脚本，实现了两种方法：忽略二次项和线性化二次项。

## 2. 数据文件

-   `station_data_input.xlsx`: 由 `generate_station_data.py` 生成的模拟输入数据，包含100个站点的ID和预测净需求量。

## 3. 优化结果文件

根据不同的优化模型，生成了两组结果：

### 方法一：忽略 $(d_i - c_i)^2$ 项

-   `result1_1_method1.xlsx`: 站点的保留/关闭状态。
-   `result1_2_method1.xlsx`: 保留站点的容量配置。

### 方法二：线性化 $(d_i - c_i)^2$ 项 (近似为 $|d_i - c_i|$)

-   `result1_1_method2.xlsx`: 站点的保留/关闭状态。
-   `result1_2_method2.xlsx`: 保留站点的容量配置。

## 4. 分析报告

-   `station_optimization_report.md`: 本次站点优化任务的综合分析报告，详细解释了模型、方法、结果和结论。

## 5. 历史文件 (来自之前任务)

这些文件与当前的站点优化任务无直接关系，是之前协作中产生的。

-   `create_sample_docs.py`, `word_files_to_merge/`, `merged_document*.docx`, `merge_word_docs*.py`: 用于合并Word文档的工具和示例。
-   `demand_forecast_data*.csv`, `generate_demand_forecast_data.py`, `demand_forecast_model*.py`, `demand_forecast_model.pkl`, `*.png`: 需求预测相关的数据、模型和结果。
-   `demand_forecast_project_report.md`, `demand_forecast_data_description.md`, `dataset_description.md`: 需求预测项目相关文档。

所有与站点优化直接相关的文件均已生成并存放在此目录中。