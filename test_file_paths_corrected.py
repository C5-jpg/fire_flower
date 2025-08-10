# -*- coding: utf-8 -*-
import pandas as pd
import os

# 使用 Unicode 转义序列来正确表示包含中文引号的目录名
# “ 是 \u201c
# ” 是 \u201d
DATA_DIR = 'C:\\Users\\hk\\fire_flower\\2025年度\u201c火花杯\u201d数学建模精英联赛-B题-附件'
DEMAND_FEATURES_FILE = os.path.join(DATA_DIR, 'demand_features.csv')
TRIPDATA_FILE = os.path.join(DATA_DIR, '202503-capitalbikeshare-tripdata.csv')

print("Checking file paths...")
print(f"DATA_DIR: {DATA_DIR}")
print(f"DEMAND_FEATURES_FILE: {DEMAND_FEATURES_FILE}")
print(f"TRIPDATA_FILE: {TRIPDATA_FILE}")

if os.path.exists(DEMAND_FEATURES_FILE):
    print("\ndemand_features.csv exists.")
    df = pd.read_csv(DEMAND_FEATURES_FILE)
    print("Head of demand_features.csv:")
    print(df.head())
else:
    print("\ndemand_features.csv NOT found.")

if os.path.exists(TRIPDATA_FILE):
    print("\n202503-capitalbikeshare-tripdata.csv exists.")
    print("File size check (using os.path.getsize):")
    size_bytes = os.path.getsize(TRIPDATA_FILE)
    size_mb = size_bytes / (1024 * 1024)
    print(f"  Size: {size_bytes} bytes ({size_mb:.2f} MB)")
else:
    print("\n202503-capitalbikeshare-tripdata.csv NOT found.")