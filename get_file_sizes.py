import os

# 附件文件夹路径 (使用原始中文引号字符)
DATA_DIR = r'C:\Users\hk\fire_flower\2025年度“火花杯”数学建模精英联赛-B题-附件'

# 列出目录下的所有文件
try:
    files = os.listdir(DATA_DIR)
    print(f"{'File Name':<40} {'Size (Bytes)':<15}")
    print("-" * 55)
    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        # 检查是否是文件（而不是目录）
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"{filename:<40} {size:<15,}")
except FileNotFoundError:
    print(f"Error: Directory not found: {DATA_DIR}")
except Exception as e:
    print(f"An error occurred: {e}")