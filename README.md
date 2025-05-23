# 스마트 팩토리 전력 예측 AI 챌린지 🏭⚡

> **전력 소비 시계열 데이터를 활용한 스마트 팩토리 전력 예측 모델 개발**

## 📋 프로젝트 개요

본 프로젝트는 공장 현장에서 발생하는 설비별 전력 소비 시계열 데이터를 바탕으로 **2025년 5월 1일 24시간의 전력 소비량을 예측**하는 AI 모델을 개발하는 것이 목표입니다.

### 🎯 평가 기준 (총 100%)
- **일별 전력 예측** (30%): MAE, RMSE, SMAPE
- **주별 전력 예측** (30%): MAE, RMSE, SMAPE  
- **5월 전체 전기요금** (20%): 총 전력량 × 180원/kWh
- **5월 전체 탄소배출량** (20%): 총 전력량 × 0.424kgCO₂/kWh

### 📊 데이터 정보
- **원본 규모**: 23,587,209 행 × 19 컬럼 (약 3.34GB)
- **처리된 샘플**: 100,000 행 × 97 컬럼 (약 107MB)
- **기간**: 2024-12-01 ~ 2025-04-29 (시간별 전력 소비 데이터)
- **주요 변수**: `activePower` (시간당 전력), `accumActiveEnergy` (누적 전력)
- **설비 정보**: `module(equipment)` - 다양한 공장 설비
- **전압 데이터**: `voltageR`, `voltageS`, `voltageT`

## 🚀 빠른 시작 (Quick Start)

### 1. 환경 설정
```bash
cd Not_Null/src/preprocessing
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. 전처리 파이프라인 실행
```bash
python main.py
```

실행 후 선택사항:
- **샘플 데이터 사용**: `y` (추천, 빠른 테스트용)
- **샘플 크기**: `100000` (10만개, 기본값)
- **분할 방식**: `1` (submission 모드, 추천)
- **데이터 저장**: `y`

### 3. 처리된 데이터 확인
```
data/processed/
├── train_data_20250523_115810.csv      # 훈련 데이터 (90,818개, 97MB)
├── val_data_20250523_115810.csv        # 검증 데이터 (9,182개, 10MB)
├── processed_features_20250523_115810.csv  # 전체 특성 데이터 (100,000개, 107MB)
└── metadata_20250523_115810.json       # 메타데이터 (1.3KB)
```

## 🛠️ 모듈 구조

```
src/preprocessing/
├── config.py              # 설정 파일 (샘플 크기: 100,000)
├── utils.py               # 유틸리티 함수
├── data_loader.py         # 데이터 로딩 (메모리 효율적)
├── feature_engineering.py # 시계열 특성 공학 (97개 특성 생성)
├── data_splitter.py       # 데이터 분할 및 평가
└── main.py               # 통합 파이프라인
```

### 🔧 주요 기능

#### 1. 메모리 효율적 데이터 로딩
```python
from data_loader import PowerDataLoader

loader = PowerDataLoader()
# 빠른 정보 확인
info = loader.get_data_info()
# 총 행 수: 23,587,209, 예상 메모리: 3.34GB

# 10만개 샘플 데이터 로딩 (최적화된 크기)
sample_data = loader.load_sample_data(100000)
# 로딩 후: (100,000, 23) → 메모리 사용량 50% 절약

# 청크 방식 전체 데이터 처리
for chunk in loader.load_chunks():
    # 청크별 처리
    pass
```

#### 2. 고급 시계열 특성 공학 (23개 → 97개 특성)
```python
from feature_engineering import PowerFeatureEngineer

engineer = PowerFeatureEngineer()
featured_data = engineer.run_feature_engineering(data)

# 생성되는 특성들 (총 97개):
# - 기본 특성: 23개
# - Lag features (8개): 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h
# - Rolling statistics (25개): 5윈도우 × 5통계량 (mean, std, min, max, median)
# - Cyclical features (8개): hour_sin/cos, day_sin/cos, month_sin/cos, year_sin/cos
# - Difference features (8개): diff_1, diff_24, diff_168, pct_change 등
# - Voltage features (6개): mean, std, range, spike detection
# - Power ratios (6개): vs 24h mean, vs weekly mean, vs hourly mean
# - Anomaly detection (4개): z-score, IQR outliers, voltage anomaly
# - Equipment features (9개): 설비별 특성 추출
```

#### 3. 시계열 데이터 분할
```python
from data_splitter import TimeSeriesSplitter

splitter = TimeSeriesSplitter()

# 제출용 분할 (추천) - 10만개 샘플 기준
train, val = splitter.create_submission_split(data)
# Train: 90,818개 (2024-12-01 ~ 2025-04-15)
# Val: 9,182개 (2025-04-15 ~ 2025-04-29)

# 또는 비율 기반 분할
train, val, test = splitter.split_by_ratio(data)
```

#### 4. 챌린지 평가 지표
```python
from data_splitter import PowerPredictionEvaluator

evaluator = PowerPredictionEvaluator()
metrics = evaluator.comprehensive_evaluation(y_true, y_pred, datetime_index)

# 자동 계산되는 지표:
# - 시간별: MAE, RMSE, SMAPE
# - 일별: MAE, RMSE, SMAPE (30% 가중치)
# - 주별: MAE, RMSE, SMAPE (30% 가중치)  
# - 5월 전기요금 (20% 가중치)
# - 5월 탄소배출량 (20% 가중치)
```

## 🤖 ML/DL 모델 개발 가이드

### 추천 모델 (10만개 샘플 최적화)
1. **LSTM/GRU**: 시계열 패턴 학습에 효과적, 97개 특성으로 풍부한 정보
2. **XGBoost/LightGBM**: 특성 공학된 데이터에 강력, 빠른 학습
3. **Transformer**: 장기 의존성 모델링, 충분한 데이터 크기
4. **ARIMA/SARIMAX**: 통계적 기준 모델

### 데이터 로딩 예시
```python
import pandas as pd

# 처리된 10만개 샘플 데이터 로딩
train_data = pd.read_csv('data/processed/train_data_20250523_115810.csv', 
                        index_col=0, parse_dates=True)
val_data = pd.read_csv('data/processed/val_data_20250523_115810.csv', 
                      index_col=0, parse_dates=True)

print(f"훈련 데이터: {train_data.shape}")  # (90818, 97)
print(f"검증 데이터: {val_data.shape}")   # (9182, 97)
print(f"메모리 사용량: {train_data.memory_usage(deep=True).sum() / 1024**2:.1f}MB")

# 특성과 타겟 분리
target_col = 'activePower'
feature_cols = [col for col in train_data.columns if col != target_col]

X_train = train_data[feature_cols]  # 96개 특성
y_train = train_data[target_col]
X_val = val_data[feature_cols]
y_val = val_data[target_col]
```

### 모델 학습 예시 (XGBoost)
```python
import xgboost as xgb
from data_splitter import PowerPredictionEvaluator

# 10만개 샘플에 최적화된 하이퍼파라미터
model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=8,          # 97개 특성에 맞게 증가
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=50,
          verbose=100)

# 예측
y_pred = model.predict(X_val)

# 평가
evaluator = PowerPredictionEvaluator()
metrics = evaluator.comprehensive_evaluation(y_val, y_pred, val_data.index)
print(f"SMAPE: {metrics['smape']:.3f}")
print(f"MAE: {metrics['mae']:.3f}")
print(f"RMSE: {metrics['rmse']:.3f}")
```

### 제출 파일 생성
```python
from data_splitter import create_submission_file

# 2025년 5월 1일 24시간 예측
# (실제로는 학습된 모델로 예측)
hourly_predictions = model.predict(may_features)  # 24개 값
agg_predictions = np.cumsum(hourly_predictions)   # 누적 값
may_bill = np.sum(hourly_predictions) * 180      # 5월 전체 전기요금
may_carbon = np.sum(hourly_predictions) * 0.424  # 5월 전체 탄소배출량

submission = create_submission_file(
    hourly_predictions, agg_predictions,
    may_bill, may_carbon,
    'submission.csv'
)
```

## 📈 특성 중요도 분석 (97개 특성)

생성된 특성들은 다음과 같이 그룹화됩니다:

```python
feature_groups = {
    'lag_features': [8개],      # 지연 특성 (1h~168h)
    'rolling_stats': [25개],    # 이동 통계 (5윈도우×5통계량)
    'cyclical': [8개],          # 사이클릭 시간 특성
    'difference': [8개],        # 차분 특성
    'voltage': [6개],           # 전압 관련 특성
    'ratios': [6개],           # 전력 비율 특성
    'anomaly': [4개],          # 이상치 탐지 특성
    'equipment': [9개],        # 설비별 특성
    'original': [23개]         # 원본 특성
}
```

### 특성 중요도 예시 분석
```python
import matplotlib.pyplot as plt

# XGBoost 특성 중요도
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

# Top 20 특성 시각화
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.title('Top 20 특성 중요도 (10만개 샘플 기준)')
plt.xlabel('중요도')
plt.tight_layout()
plt.show()
```

## 🔍 성능 최적화 팁 (10만개 샘플)

### 1. 메모리 최적화 (이미 적용됨)
```python
# 데이터 타입 최적화로 메모리 50% 절약
# float64 → float32: 메모리 50% 절약
# int64 → int32: 메모리 50% 절약
# object → category: 메모리 대폭 절약
# 10만개 × 97개 특성 = 약 56MB 메모리 사용
```

### 2. 시계열 교차 검증 (10만개 샘플 최적화)
```python
from sklearn.model_selection import TimeSeriesSplit

# 10만개 샘플에 적합한 분할 수
tscv = TimeSeriesSplit(n_splits=5)  # 각 fold: ~18,000개 샘플
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    print(f"Fold {fold+1}: Train {len(train_idx)}, Val {len(val_idx)}")
    # 모델 학습 및 검증
```

### 3. 하이퍼파라미터 튜닝 (10만개 최적화)
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 6, 12),        # 97개 특성 고려
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
    }
    
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)], 
              early_stopping_rounds=50, 
              verbose=0)
    
    y_pred = model.predict(X_val)
    evaluator = PowerPredictionEvaluator()
    smape = evaluator.smape(y_val, y_pred)
    return smape

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print(f"Best SMAPE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### 4. 앙상블 모델 (10만개 샘플)
```python
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor

# 10만개 샘플로 충분한 앙상블 학습
models = [
    ('xgb', xgb.XGBRegressor(n_estimators=1000, max_depth=8)),
    ('cat', CatBoostRegressor(iterations=1000, depth=8, verbose=False)),
    ('ridge', Ridge(alpha=1.0))
]

ensemble = VotingRegressor(models)
ensemble.fit(X_train, y_train)

# 앙상블 예측
y_pred_ensemble = ensemble.predict(X_val)
```

## ❗ 주의사항 (10만개 샘플 특화)

### 1. 시간 순서 준수
- **절대** 미래 데이터가 과거 예측에 사용되지 않도록 주의
- 특성 생성 시 lag와 rolling window 고려 (최대 168h)
- 데이터 분할: Train (90,818개) → Val (9,182개)

### 2. 메모리 관리 (최적화됨)
- 10만개 샘플: 약 107MB (관리 가능한 크기)
- 97개 특성: 메모리 효율적으로 최적화됨
- 전체 23M 데이터는 필요시에만 사용

### 3. 타겟 변수
- `activePower`: 시간당 전력 소비량 (kW) - **주요 타겟**
- 범위: 890~5,190 kW, 평균: 3,011 kW
- `accumActiveEnergy`: 누적 전력 소비량 (kWh)

### 4. 결측치 및 이상치 (처리됨)
- 전처리 단계에서 2,397개 결측치 처리됨
- 이상치 탐지 특성 포함 (z-score, IQR)
- 새로운 결측치 발생 시 forward fill → backward fill 적용

## 🏆 최종 체크리스트 (10만개 샘플)

### 모델 개발 전
- [ ] 10만개 샘플 데이터 로딩 및 전처리 완료 ✅
- [ ] 97개 특성 공학 결과 확인 ✅
- [ ] 훈련(90,818)/검증(9,182) 데이터 분할 확인 ✅
- [ ] 타겟 변수 분포 분석 (평균: 3,011 kW)

### 모델 학습 중
- [ ] 시계열 교차 검증 적용 (5-fold 권장)
- [ ] 평가 지표 모니터링 (MAE, RMSE, SMAPE)
- [ ] 과적합 방지 (early stopping, 97개 특성 고려)
- [ ] 특성 중요도 분석 (그룹별 기여도)

### 제출 전
- [ ] 2025-05-01 24시간 예측 생성
- [ ] submission 파일 형식 확인
- [ ] 예측값 범위 검증 (890~5,190 kW 내)
- [ ] 5월 전기요금, 탄소배출량 계산 확인

## 📊 데이터 통계 (10만개 샘플)

```
데이터 개요:
- 총 샘플: 100,000개
- 총 특성: 97개 (23개 → 97개로 확장)
- 시간 범위: 2024-12-01 ~ 2025-04-29
- 주파수: 시간별 (hourly)
- 메모리 사용량: 56.3MB

훈련/검증 분할:
- 훈련 데이터: 90,818개 (90.8%)
- 검증 데이터: 9,182개 (9.2%)
- 예측 시작: 2025-05-01 00:00:00

타겟 변수 통계:
- 평균: 3,011 kW
- 표준편차: 789 kW
- 최소값: 890 kW
- 최대값: 5,190 kW
- 중앙값: 3,045 kW
```

## 📞 문의 및 지원

- **이슈 트래킹**: GitHub Issues
- **코드 리뷰**: Pull Request
- **팀 회의**: 주간 진행 상황 공유

---

## 📁 파일 구조

```
Not_Null/
├── data/
│   ├── raw/                    # 원본 데이터
│   │   ├── train.csv          # 훈련 데이터 (23,587,209 행, 3.34GB)
│   │   ├── test.csv           # 테스트 데이터
│   │   └── sample_submission_final.csv
│   └── processed/              # 전처리된 데이터 (10만개 샘플)
│       ├── train_data_20250523_115810.csv      # 90,818개, 97MB
│       ├── val_data_20250523_115810.csv        # 9,182개, 10MB  
│       ├── processed_features_20250523_115810.csv # 100,000개, 107MB
│       └── metadata_20250523_115810.json       # 메타데이터, 1.3KB
├── src/
│   └── preprocessing/          # 전처리 모듈 (97개 특성 생성)
├── notebooks/                  # EDA 노트북
├── reports/                    # 결과 보고서
└── README.md                   # 이 파일
```

**10만개 샘플로 성공적인 모델 개발을 응원합니다! 🍀**