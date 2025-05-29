# 스마트 팩토리 전력 예측 - ML/DL 데이터 준비 가이드

## 📋 목차
- [1. 개요](#1-개요)
- [2. 데이터 구조 설계](#2-데이터-구조-설계)
- [3. 시간 피처 엔지니어링](#3-시간-피처-엔지니어링)
- [4. Tabular vs Time Series 분리](#4-tabular-vs-time-series-분리)
- [5. NPY 파일 형식 선택](#5-npy-파일-형식-선택)
- [6. Test 데이터 처리](#6-test-데이터-처리)
- [7. 파일 구조 및 사용법](#7-파일-구조-및-사용법)
- [8. 문제점 및 해결방안](#8-문제점-및-해결방안)

---

## 1. 개요

### 🎯 목표
스마트 팩토리 전력 소비 데이터를 다양한 ML/DL 알고리즘에 적합한 형태로 변환하여 정확한 전력 예측 모델 개발을 지원

### 📊 처리 대상
- **Train 데이터**: `train.csv` (3.2GB) → 전처리 → 학습/검증용 분할
- **Test 데이터**: `test.csv` (1.4GB) → 예측용 데이터
- **Submission**: `sample_submission_final.csv` → 2025-05-01 24시간 예측

### 🏭 도메인 특성
- **시간 의존성**: 시간대별 전력 소비 패턴이 명확
- **주기성**: 일일/주간/계절별 반복 패턴
- **운영 특성**: 업무시간, 주말, 야간교대 등 공장 운영 패턴

---

## 2. 데이터 구조 설계

### 📈 최종 데이터 형태

#### Train/Validation 분할
```
Train 데이터 (train.csv 기반):
├── X_train: 80,000 샘플 (80%)
├── y_train: 80,000 샘플
├── X_val: 20,000 샘플 (20%)
└── y_val: 20,000 샘플

Test 데이터 (test.csv 기반):
├── X_test: 24 샘플 (submission 요구사항)
└── test_data: 24 샘플 (메타데이터 포함)
```

#### 핵심 피처 선택 (17개)
```python
전력 관련 (11개):
- voltageR/S/T (3상 전압)
- currentR/S/T (3상 전류) 
- powerFactorR/S/T (3상 역률)
- accumActiveEnergy (누적 에너지)
- operation (운영 상태)

시간 관련 (6개):
- hour, day_of_week, month
- is_weekend, is_peak_hour, is_business_hour
```

### 🎯 설계 원칙
1. **단순성**: 핵심 피처만 선택하여 과적합 방지
2. **도메인 지식**: 스마트 팩토리 운영 특성 반영
3. **호환성**: 다양한 ML/DL 알고리즘 지원
4. **효율성**: 빠른 로딩과 학습을 위한 최적화

---

## 3. 시간 피처 엔지니어링

```
📊 시간대별 전력 패턴:
- 08:00-18:00: 피크 시간 (최대 전력)
- 09:00-17:00: 업무 시간 (안정적 고전력)
- 18:00-08:00: 야간 시간 (최소 전력)
- 주말: 대폭 감소
- 계절별: 냉난방 부하 변화
```

### 🔧 생성된 시간 피처들

#### 1. 기본 시간 피처 (6개)
```python
hour          # 0-23 (시간대별 패턴)
day_of_week   # 0-6 (평일/주말 패턴)
day_of_month  # 1-31 (월내 패턴)
month         # 1-12 (계절별 패턴)
quarter       # 1-4 (분기별 패턴)
year          # 연도별 변화
```

#### 2. 순환 인코딩 피처 (6개)
```python
hour_sin = sin(2π × hour/24)      # 시간의 순환성
hour_cos = cos(2π × hour/24)      # 23시와 0시가 가까움
day_sin = sin(2π × day/7)         # 요일의 순환성
day_cos = cos(2π × day/7)         # 일요일과 월요일이 가까움
month_sin = sin(2π × month/12)    # 계절의 순환성
month_cos = cos(2π × month/12)    # 12월과 1월이 가까움
```

#### 3. 비즈니스 로직 피처 (4개)
```python
is_weekend      # 주말 여부 (전력 급감)
is_business_hour # 업무시간 (09:00-17:00, 안정적 고전력)
is_peak_hour    # 피크시간 (08:00-18:00, 최대 전력)
is_night_shift  # 야간교대 (전력 패턴 변화)
```

### 💡 순환 인코딩이 필요한 이유

**문제**: 시간을 단순 숫자로 표현하면 23시와 0시가 멀어 보임
```python
hour = [22, 23, 0, 1]  # 23과 0이 실제로는 1시간 차이인데 23만큼 떨어져 보임
```

**해결**: Sin/Cos 변환으로 순환성 표현
```python
# 23시와 0시가 수학적으로 가까운 값으로 표현됨
hour_sin[23] ≈ hour_sin[0]
hour_cos[23] ≈ hour_cos[0]
```

---

## 4. Tabular vs Time Series 분리

### 🤔 왜 두 가지 형태로 나눴는가?

**서로 다른 알고리즘이 서로 다른 데이터 구조를 요구하기 때문에.**

### 📋 Tabular 데이터 (일반 ML용)

#### 구조
```python
X_train.shape: (80000, 17)  # (샘플수, 피처수)
y_train.shape: (80000,)     # (샘플수,)

# 각 행은 독립적인 시점의 데이터
X_train[0] = [voltage_R, voltage_S, ..., hour=9, is_weekend=0, ...]
y_train[0] = 45.2  # 해당 시점의 전력값
```

#### 적용 모델
- **RandomForest**: 트리 기반, 피처 중요도 분석
- **XGBoost**: 그래디언트 부스팅, 높은 성능
- **LightGBM**: 빠른 학습, 메모리 효율적
- **SVM**: 서포트 벡터 머신
- **Linear Regression**: 선형 관계 분석

#### 장점
- ✅ 빠른 학습 속도
- ✅ 해석 가능성 높음
- ✅ 피처 중요도 분석 가능
- ✅ 안정적인 성능

### ⏰ Time Series 데이터 (딥러닝용)

#### 구조
```python
X_train.shape: (79981, 24, 17)  # (샘플수, 시퀀스길이, 피처수)
y_train.shape: (79981, 1)       # (샘플수, 예측값)

# 각 샘플은 24시간 연속 시퀀스
X_train[0] = [
    [시점1: voltage_R, ..., hour=1, ...],  # 1시간 전
    [시점2: voltage_R, ..., hour=2, ...],  # 2시간 전
    ...
    [시점24: voltage_R, ..., hour=24, ...] # 24시간 전
]
y_train[0] = 47.8  # 25번째 시점 예측값
```

#### 적용 모델
이건 뭐 잘 알아서 판단하리라 생각함

---

## 5. NPY 파일 형식 선택

-> 빨라서


---

## 6. Test 데이터 처리

### 🎯 Test 데이터 처리 전략

#### 원본 데이터 분석
```python
test.csv 정보:
├── 크기: 1.4GB
├── 샘플 수: ~500,000개
├── 시간 범위: 2024-12-01 ~ 2025-04-29
└── submission 요구: 2025-05-01 예측
```


#### 3. Submission 매핑
```python
# 시간 매칭 실패 시 대안
if len(matched_data) == 0:
    # 최신 24개 데이터 사용
    test_data = test_data.tail(24)
    
# Submission ID 강제 매핑
test_data['submission_id'] = submission_template['id'].values
```


## 파일 구조 및 사용법

### 📁 생성된 파일 구조

```
data/ml_ready/
├── tabular/
│   ├── X_train_20250523_153654.npy      # 학습용 피처 (80k × 17)
│   ├── y_train_20250523_153654.npy      # 학습용 타겟 (80k)
│   ├── X_val_20250523_153654.npy        # 검증용 피처 (20k × 17)
│   ├── y_val_20250523_153654.npy        # 검증용 타겟 (20k)
│   ├── X_test_20250523_153654.npy       # 테스트 피처 (24 × 17) - Test.csv 에서 따옴
│   ├── test_data_20250523_153654.npy    # 테스트 메타데이터 (24 × 26)
│   └── metadata_20250523_153654.json    # 데이터 정보
│
└── time_series/
    ├── X_train_20250523_153708.npy      # 학습용 시퀀스 (79k × 24 × 17)
    ├── y_train_20250523_153708.npy      # 학습용 타겟 (79k × 1)
    ├── X_val_20250523_153708.npy        # 검증용 시퀀스 (20k × 24 × 17)
    ├── y_val_20250523_153708.npy        # 검증용 타겟 (20k × 1)
    ├── X_test_20250523_153708.npy       # 테스트 시퀀스 (0) - Test.csv 에서 따옴
    ├── test_data_20250523_153708.npy    # 테스트 메타데이터 (24 × 26)
    └── metadata_20250523_153708.json    # 데이터 정보
```

### 📋 메타데이터 정보

#### tabular/metadata.json
```json
{
  "data_type": "tabular",
  "target_column": "activePower",
  "feature_columns": ["voltageR", "voltageS", ..., "is_business_hour"],
  "shapes": {
    "X_train": [80000, 17],
    "X_test": [24, 17]
  },
  "total_features": 17
}
```

#### time_series/metadata.json
```json
{
  "data_type": "time_series", 
  "shapes": {
    "X_train": [79981, 24, 17],  # (샘플, 시퀀스, 피처)
    "X_test": [0]                # 시퀀스 변환 안함 (당연히 없으니까)
  }
}
```

### 🚀 사용법

#### 1. Tabular 데이터 로딩 (일반 ML)
```python
import numpy as np

# 데이터 로딩
X_train = np.load('data/ml_ready/tabular/X_train_20250523_153654.npy')
y_train = np.load('data/ml_ready/tabular/y_train_20250523_153654.npy')
X_test = np.load('data/ml_ready/tabular/X_test_20250523_153654.npy')

# RandomForest 예시
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 2. Time Series 데이터 로딩 (딥러닝)
```python
import numpy as np
import torch

# 데이터 로딩
X_train = np.load('data/ml_ready/time_series/X_train_20250523_153708.npy')
y_train = np.load('data/ml_ready/time_series/y_train_20250523_153708.npy')

# PyTorch 변환
X_train = torch.FloatTensor(X_train)  # (79981, 24, 17)
y_train = torch.FloatTensor(y_train)  # (79981, 1)

# LSTM 예시
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=17, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 마지막 시점 출력
```

#### 3. Submission 생성
```python
# 예측 수행
predictions = model.predict(X_test)  # [24개 예측값]

# Submission 형태로 변환
submission = pd.read_csv('data/raw/sample_submission_final.csv')
submission['activePower'] = predictions

# 제출 파일 저장
submission.to_csv('submission_final.csv', index=False)
```

-> 이건 혹시 모르니까 GPT 한테 만들어달라고 함
---

## 📝 결론

### ✅ 성과
1. **다양한 알고리즘 지원**: Tabular(ML) + Time Series(DL)
2. **최적화된 성능**: NPY 형식으로 20-60배 빠른 로딩
3. **도메인 특화**: 스마트 팩토리 특성을 반영한 피처 엔지니어링
4. **즉시 사용 가능**: 바로 모델 학습에 투입 가능한 형태

### 🎯 사용 가이드라인
1. **빠른 프로토타입**: `tabular/` 데이터로 RandomForest, XGBoost 실험
2. **고성능 모델**: `time_series/` 데이터로 LSTM, Transformer 개발
3. **최종 앙상블**: 두 방식의 결과를 결합하여 성능 극대화

### 🚀 다음 단계
1. **모델 개발**: 준비된 데이터로 베이스라인 모델 구축
2. **성능 최적화**: 하이퍼파라미터 튜닝 및 피처 선택
3. **앙상블**: 다양한 모델의 예측 결과 결합
4. **제출**: Submission 형태로 최종 예측 결과 생성

---

