# 问题二：鲁棒优化项目文件清单与说明

此目录 `fire_flower/qwencli/qs2` 包含了完成“鲁棒优化”任务的所有相关文件。

## 1. 核心文档

-   `robust_optimization_model_design.md`: 鲁棒优化模型的详细设计文档，包括问题分析、不确定性建模、数学模型构建思路和求解策略。
-   `robust_optimization_report.md`: 本次鲁棒优化任务的综合分析报告，详细解释了模型、求解过程、结果和结论。

## 2. 核心代码文件

-   `generate_uncertainty_scenarios.py`: 用于生成模拟的不确定性场景数据的Python脚本。
-   `robust_optimization_model.py`: 鲁棒优化模型的实现脚本，使用场景近似法求解。

## 3. 数据文件

-   `uncertainty_scenarios.xlsx`: 由 `generate_uncertainty_scenarios.py` 生成的包含5个不确定性场景的模拟输入数据。

## 4. 优化结果文件

-   `result2_1.xlsx`: 鲁棒优化得到的站点保留/关闭状态。
-   `result2_2.xlsx`: 鲁棒优化得到的保留站点的容量配置。

所有与问题二相关的文件均已生成并存放在此目录中。