import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# (1) 데이터 불러오기 (.npy 기반)
X_train = np.load("data/ml_ready/tabular/X_train_20250528_205352.npy")
y_train = np.load("data/ml_ready/tabular/y_train_20250528_205352.npy")
X_val = np.load("data/ml_ready/tabular/X_val_20250528_205352.npy")
y_val = np.load("data/ml_ready/tabular/y_val_20250528_205352.npy")
X_test = np.load("data/ml_ready/tabular/X_test_20250528_205352.npy") 

# (2) 모델 학습
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    objective='reg:squarederror',
    random_state=42
)
model.fit(X_train, y_train_scaled)

# (3) 672시간 예측 (28일 × 24시간)
hourly_predictions = []

for _ in range(28):
    preds = model.predict(X_test)
    hourly_predictions.extend(preds)

# (4) agg_pow, may_bill, may_carbon 계산
agg_pow = float(np.sum(hourly_predictions))
may_bill = agg_pow * 180
may_carbon = agg_pow * 0.424

# (5) datetime index 생성
start_time = datetime(2025, 5, 1, 0, 0, 0)
datetime_ids = [start_time + timedelta(hours=i) for i in range(672)]

# (6) 제출 파일 생성
submission = pd.DataFrame({
    "id": datetime_ids,
    "hourly_pow": np.round(hourly_predictions, 4),
    "agg_pow": [round(agg_pow, 4)] * 672,
    "may_bill": [round(may_bill, 4)] * 672,
    "may_carbon": [round(may_carbon, 4)] * 672
})
submission.to_csv("submission.csv", index=False)
print("✅ 최종 제출 파일 'submission.csv' 생성 완료!")


# (1) 검증 데이터에 대한 예측
y_val_pred = model.predict(X_val)

# (2) 평가 지표 계산
mae = mean_absolute_error(y_val, y_val_pred)
#rmse = mean_squared_error(y_val, y_val_pred, squared=False)
smape = 100 * np.mean(
    2 * np.abs(y_val - y_val_pred) / (np.abs(y_val) + np.abs(y_val_pred) + 1e-8)
)

print(f"✅ Validation MAE   : {mae:.4f}")

# Plot y_val distribution
plt.figure(figsize=(6, 4))
plt.hist(y_val, bins=30, color='skyblue', edgecolor='black')
plt.title("y_val Distribution")
plt.xlabel("activePower")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("/Not_Null/y_val_distribution.png")

# Plot hourly_pred distribution
plt.figure(figsize=(6, 4))
plt.hist(hourly_predictions, bins=30, color='salmon', edgecolor='black')
plt.title("hourly_pred Distribution")
plt.xlabel("Predicted Power")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("/Not_Null/hourly_pred_distribution.png")




