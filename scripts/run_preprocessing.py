import os
import sys

# 设置正确的文件路径
script_path = os.path.join(
    r'c:\Users\hk\fire_flower\2025年度“火花杯”数学建模精英联赛-B题-附件\GLM4.5V1',
    '数据预处理.py'
)

# 执行脚本
print(f'正在运行脚本: {script_path}')
with open(script_path, 'r', encoding='utf-8') as f:
    code = f.read()

exec(code)