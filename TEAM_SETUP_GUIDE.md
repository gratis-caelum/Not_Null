# 🚀 팀원 프로젝트 설정 가이드

> **스마트 팩토리 전력 예측 프로젝트 - Not_Null 팀**

## 📋 목차
- [1. 프로젝트 클론 및 환경 설정](#1-프로젝트-클론-및-환경-설정)
- [2. 원본 데이터 다운로드](#2-원본-데이터-다운로드)
- [3. ML/DL 데이터 생성](#3-mldl-데이터-생성)
- [4. 데이터 공유 방법](#4-데이터-공유-방법)
- [5. 문제 해결](#5-문제-해결)

---

## 1. 프로젝트 클론 및 환경 설정

### 📥 Git 클론
```bash
git clone https://github.com/gratis-caelum/Not_Null.git
cd Not_Null
git checkout develop  # 개발 브랜치로 이동
```

### 🐍 Python 환경 설정
```bash
# Conda 환경 생성 (권장)
conda create -n smart_factory python=3.11
conda activate smart_factory

# 필수 패키지 한번에 설치 ⭐ 추천
pip install -r requirements.txt

# 또는 개별 설치
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm  # 고성능 ML
pip install torch torchvision  # PyTorch (딥러닝용, 선택적)
```

### ✅ 설치 확인
```python
# Python에서 테스트
import pandas as pd
import numpy as np
import sklearn
print("✅ 환경 설정 완료!")
```

---

## 2. 원본 데이터 다운로드

### 📊 필요한 데이터 파일들
프로젝트 루트의 `data/raw/` 폴더에 다음 파일들을 넣어주세요:

```
Not_Null/
└── data/
    └── raw/
        ├── train.csv (3.2GB) ⭐ 필수
        ├── test.csv (1.4GB) ⭐ 필수
        └── sample_submission_final.csv (1KB) ⭐ 필수
```

### 💾 데이터 다운로드 방법
1. **대회 사이트에서 다운로드**
2. **팀장/팀원으로부터 공유 받기**
3. **클라우드 스토리지 링크 이용**

⚠️ **주의**: 원본 데이터는 용량이 크므로 Git에 커밋하지 마세요! (`.gitignore`에 포함됨)

---

## 3. ML/DL 데이터 생성

### 🎯 **이게 핵심입니다!** 각자 데이터를 생성해야 해요

원본 데이터를 받았으면, 다음 스크립트들을 **순서대로** 실행하세요:

### 1단계: 데이터 전처리
```bash
cd src/Preprocessing
python run_preprocessing.py
```
- **소요시간**: 약 5-10분
- **결과**: `data/processed/preprocessed_data_YYYYMMDD_HHMMSS.csv`
- **내용**: 이상치 제거, 시간 피처 추가, 스케일링

### 2단계: ML/DL 데이터 준비
```bash
python prepare_ml_data.py
```
- **소요시간**: 약 2-5분  
- **사용자 선택**:
  ```
  🔧 ML/DL 데이터 타입을 선택하세요:
     1. 테이블 형태 (일반 ML 모델용)
     2. 시계열 시퀀스 (LSTM, GRU 등 DL 모델용)  
     3. 둘 다 생성 ⭐ 추천
  
  선택 (1-3): 3
  ```

### 3단계: 생성 결과 확인
```
data/ml_ready/
├── tabular/          ← RandomForest, XGBoost용
│   ├── X_train_*.npy (80k samples × 17 features)
│   ├── X_val_*.npy   (20k samples × 17 features)
│   ├── X_test_*.npy  (24 samples × 17 features)
│   └── metadata_*.json
└── time_series/      ← LSTM, Transformer용  
    ├── X_train_*.npy (79k samples × 24 timesteps × 17 features)
    ├── X_val_*.npy   (20k samples × 24 timesteps × 17 features)
    └── metadata_*.json
```

---

## 4. 데이터 공유 방법

### 🤔 **왜 Git에 올리지 않나요?**
- **대용량**: ML 데이터는 100-250MB로 GitHub 한계 초과
- **개인화**: 각자 다른 전처리 실험을 할 수 있음  
- **효율성**: 코드만 공유하고 각자 생성하는 것이 더 빠름

### ✅ **권장 방식: 코드 공유 + 각자 생성**

#### 장점:
- ✅ **빠른 동기화**: Git pull만 하면 최신 코드 받기
- ✅ **실험 자유도**: 각자 다른 파라미터로 실험 가능
- ✅ **저장소 용량**: GitHub 저장소가 가벼움
- ✅ **재현성**: 동일한 코드로 동일한 결과 보장

#### 팀워크 플로우:
```bash
# 1. 최신 코드 받기  
git pull origin develop

# 2. 데이터 생성 (각자)
python src/Preprocessing/run_preprocessing.py
python src/Preprocessing/prepare_ml_data.py

# 3. 모델 개발 및 결과 공유
python your_model.py
git add your_model.py
git commit -m "feat: RandomForest 베이스라인 모델 추가"
git push origin develop
```

### 📤 **대안: 클라우드 공유 (필요시)**

급하게 데이터를 공유해야 할 때만:

```bash
# Google Drive / Dropbox 등에 압축 업로드
zip -r ml_data.zip data/ml_ready/
# → 클라우드에 업로드 후 링크 공유
```

---

## 5. 문제 해결

### ❌ **자주 발생하는 문제들**

#### 1. 원본 데이터 없음
```
FileNotFoundError: train.csv를 찾을 수 없습니다
```
**해결**: `data/raw/` 폴더에 `train.csv`, `test.csv` 파일 확인

#### 2. 메모리 부족
```
MemoryError: 메모리가 부족합니다
```
**해결**: `run_preprocessing.py`에서 샘플 크기 줄이기
```python
# 기본: 100,000 → 50,000으로 변경
sample_size = 50000
```

#### 3. 패키지 없음
```
ModuleNotFoundError: No module named 'sklearn'
```
**해결**: 필수 패키지 설치
```bash
pip install -r requirements.txt  # 한번에 설치
# 또는
pip install pandas numpy scikit-learn matplotlib seaborn
```

#### 4. 시계열 데이터 생성 실패
```
X_test shape: (0,) # 비어있음
```
**해결**: 정상입니다! 테스트 데이터가 24개뿐이라 시퀀스 생성 불가
- Tabular 데이터 사용하거나
- 시퀀스 길이를 12로 줄이기

### 💡 **성능 최적화 팁**

#### SSD 사용 권장
```bash
# 데이터 처리 속도 비교
HDD: 10-15분
SSD: 3-5분  ⭐ 권장
```

#### 메모리 16GB 이상 권장
```python
# 샘플 크기별 메모리 사용량
50k samples: ~4GB
100k samples: ~8GB  ⭐ 권장
200k samples: ~16GB
```

---

## 📞 도움 요청

### 🆘 **문제 해결이 안 될 때**

1. **GitHub Issues 활용**
   ```
   제목: [HELP] 데이터 전처리 실패
   내용: 
   - 환경: macOS/Windows
   - 에러 메시지: (전체 복사)
   - 시도한 방법: ...
   ```

2. **팀 채팅방에서 질문**
   - 스크린샷과 함께
   - 에러 메시지 전체 복사

3. **페어 프로그래밍**
   - 화면 공유로 함께 해결
   - Cursor 등 협업 툴 활용

---

## 🎯 **성공 체크리스트**

다음이 모두 완료되면 준비 완료! ✅

```bash
□ Git 클론 완료
□ Python 환경 설정 완료  
□ 원본 데이터 (train.csv, test.csv) 준비
□ 전처리 스크립트 실행 성공
□ ML/DL 데이터 생성 완료
□ data/ml_ready/ 폴더에 .npy 파일들 확인
□ 첫 번째 모델 실험 준비 완료!
```

### 🚀 **다음 단계**
1. **베이스라인 모델**: RandomForest 또는 XGBoost
2. **성능 개선**: 피처 엔지니어링, 하이퍼파라미터 튜닝  
3. **앙상블**: 여러 모델 조합으로 성능 극대화
4. **제출**: Submission 파일 생성

---

*문서 작성일: 2025-05-23*  
*작성자: Not_Null 팀* 

-> GPT 짱!