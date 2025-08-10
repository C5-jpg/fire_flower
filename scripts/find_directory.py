import os

# 使用相对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"BASE_DIR: {BASE_DIR}")

# 列出BASE_DIR下的所有文件和文件夹
print("\nBASE_DIR 下的文件和文件夹:")
for item in os.listdir(BASE_DIR):
    item_path = os.path.join(BASE_DIR, item)
    if os.path.isdir(item_path):
        print(f"  [DIR] {item}")
    else:
        print(f"  [FILE] {item}")

# 尝试找到包含"火花杯"的目录
print("\n尝试找到包含'火花杯'的目录:")
for item in os.listdir(BASE_DIR):
    if "火花杯" in item and os.path.isdir(os.path.join(BASE_DIR, item)):
        print(f"  找到匹配的目录: {item}")
        # 列出该目录下的文件
        sub_dir = os.path.join(BASE_DIR, item)
        print(f"  {item} 下的文件和文件夹:")
        for sub_item in os.listdir(sub_dir):
            sub_item_path = os.path.join(sub_dir, sub_item)
            if os.path.isdir(sub_item_path):
                print(f"    [DIR] {sub_item}")
            else:
                print(f"    [FILE] {sub_item}")