import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 加载数据 ---
print("正在加载数据...")
df = pd.read_csv('demand_forecast_data.csv', encoding='utf-8-sig')
print(f"数据加载完成，共 {len(df)} 行。")

# --- 2. 数据预处理 ---
print("正在进行数据预处理...")

# 处理日期特征
df['date'] = pd.to_datetime(df['date'])
# 可以提取更多时间特征
df['day_of_year'] = df['date'].dt.dayofyear
df['quarter'] = df['date'].dt.quarter

# 对分类变量进行One-Hot编码
# 注意：One-Hot编码后，原始的 'poi_id' 列会被移除，替换为 'poi_id_X' 列
categorical_features = ['poi_id', 'holiday_name', 'weather_condition']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# 处理缺失值 (主要针对历史需求特征)
# 使用前向填充或插值法填充历史需求特征
# 首先确保数据按 'date' 排序
df_encoded = df_encoded.sort_values(by=['date']).reset_index(drop=True)

# 重新获取One-Hot编码后的POI列名
poi_columns = [col for col in df_encoded.columns if col.startswith('poi_id_')]

# 对于历史需求特征的缺失值，我们采用简单的前向填充方法
# 这在时间序列数据中是常用且合理的
df_encoded['historical_demand_lag_1'] = df_encoded['historical_demand_lag_1'].fillna(method='ffill')
df_encoded['historical_demand_lag_7'] = df_encoded['historical_demand_lag_7'].fillna(method='ffill')
df_encoded['historical_demand_rolling_mean_7'] = df_encoded['historical_demand_rolling_mean_7'].fillna(method='ffill')

# 再次检查并处理任何剩余的缺失值 (用均值填充)
for col in df_encoded.columns:
    if df_encoded[col].isnull().any():
        if df_encoded[col].dtype != 'object': # 只处理数值型特征
            df_encoded[col].fillna(df_encoded[col].mean(), inplace=True)

# 定义特征和目标变量
# 移除不需要的列
drop_cols = ['date', 'demand'] 
X = df_encoded.drop(drop_cols, axis=1)
y = df_encoded['demand']

print(f"特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# --- 3. 划分训练集和测试集 ---
# 按时间顺序划分，避免数据泄露
# 假设数据已按日期排序
split_date = '2023-10-01'
train_mask = df_encoded['date'] < split_date
test_mask = df_encoded['date'] >= split_date

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# --- 4. 模型训练 ---
print("正在训练随机森林模型...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("模型训练完成。")

# --- 5. 模型预测与评估 ---
print("正在对测试集进行预测...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- 模型评估结果 ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.4f}")

# --- 6. 特征重要性 ---
print("\n正在分析特征重要性...")
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
print(feature_importances.head(10))

# --- 7. 保存模型 ---
print("\n正在保存模型...")
joblib.dump(model, 'demand_forecast_model.pkl')
print("模型已保存为 'demand_forecast_model.pkl'。")

# --- 8. 可视化结果 ---
# 1. 真实值 vs 预测值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实需求量')
plt.ylabel('预测需求量')
plt.title('真实需求量 vs 预测需求量')
plt.savefig('true_vs_pred.png')
plt.close()

# 2. 残差图
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='dashed')
plt.xlabel('预测需求量')
plt.ylabel('残差 (真实 - 预测)')
plt.title('残差图')
plt.savefig('residuals.png')
plt.close()

# 3. 特征重要性 Top 10
plt.figure(figsize=(10, 6))
top_features = feature_importances.head(10)
sns.barplot(x='importance', y='feature', data=top_features)
plt.title('特征重要性 (Top 10)')
plt.savefig('feature_importance.png')
plt.close()

print("\n所有图表已保存。")
print("\n需求预测模型训练和评估完成。")