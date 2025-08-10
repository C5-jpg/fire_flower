# -*- coding: utf-8 -*-
import pandas as pd
import os

DATA_DIR = 'C:\\Users\\hk\\fire_flower\\2025年度"火花杯"数学建模精英联赛-B题-附件'
DEMAND_FEATURES_FILE = os.path.join(DATA_DIR, 'demand_features.csv')
TRIPDATA_FILE = os.path.join(DATA_DIR, '202503-capitalbikeshare-tripdata.csv')

print("Checking file paths...")
print(f"DEMAND_FEATURES_FILE: {DEMAND_FEATURES_FILE}")
print(f"TRIPDATA_FILE: {TRIPDATA_FILE}")

if os.path.exists(DEMAND_FEATURES_FILE):
    print("demand_features.csv exists.")
    df = pd.read_csv(DEMAND_FEATURES_FILE)
    print(df.head())
else:
    print("demand_features.csv NOT found.")

if os.path.exists(TRIPDATA_FILE):
    print("202503-capitalbikeshare-tripdata.csv exists.")
else:
    print("202503-capitalbikeshare-tripdata.csv NOT found.")