import os

# 使用相对路径 (使用正确的目录名)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '2025年度"火花杯"数学建模精英联赛-B题-附件')
TRIPDATA_FILE = os.path.join(DATA_DIR, '202503-capitalbikeshare-tripdata.csv')
DEMAND_FEATURES_FILE = os.path.join(DATA_DIR, 'demand_features.csv')
METRO_FILE = os.path.join(DATA_DIR, 'metro_coverage.xlsx')

print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"TRIPDATA_FILE: {TRIPDATA_FILE}")
print(f"DEMAND_FEATURES_FILE: {DEMAND_FEATURES_FILE}")
print(f"METRO_FILE: {METRO_FILE}")

# 检查文件是否存在
print(f"\n检查文件是否存在:")
print(f"TRIPDATA_FILE exists: {os.path.exists(TRIPDATA_FILE)}")
print(f"DEMAND_FEATURES_FILE exists: {os.path.exists(DEMAND_FEATURES_FILE)}")
print(f"METRO_FILE exists: {os.path.exists(METRO_FILE)}")

# 尝试列出DATA_DIR下的文件
if os.path.exists(DATA_DIR):
    print(f"\n{DATA_DIR} 下的文件和文件夹:")
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path):
            print(f"  [DIR] {item}")
        else:
            print(f"  [FILE] {item}")
else:
    print(f"\n目录 {DATA_DIR} 不存在")