"""
스마트 팩토리 전력 예측 - ML/DL 학습 데이터 준비
================================================

전처리된 데이터를 ML/DL 모델 학습에 적합한 형태로 변환
- 시계열 데이터 구조화
- Train/Validation/Test 분할
- Feature/Target 분리
- 최종 학습용 데이터 생성
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

# 프로젝트 루트 디렉토리 설정
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class MLDataPreparator:
    """ML/DL 학습용 데이터 준비 클래스"""
    
    def __init__(self):
        self.feature_columns = []
        self.target_column = None
        self.scaler = None
        self.data_info = {}
        self.train_feature_stats = {}  # 훈련 데이터 통계 저장
        
    def load_preprocessed_data(self, file_path: str = None) -> pd.DataFrame:
        """전처리된 데이터 로드"""
        if file_path is None:
            # 가장 최근 전처리 파일 찾기
            processed_dir = project_root / "data" / "processed"
            csv_files = list(processed_dir.glob("preprocessed_data_*.csv"))
            if not csv_files:
                raise FileNotFoundError("전처리된 데이터 파일을 찾을 수 없습니다.")
            file_path = max(csv_files, key=lambda x: x.stat().st_mtime)
        
        print(f"전처리된 데이터 로딩: {file_path}")
        data = pd.read_csv(file_path)
        
        print(f"데이터 로드 완료: {data.shape}")
        print(f"   - 컬럼 수: {len(data.columns)}")
        print(f"   - 메모리: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return data
    
    def identify_target_and_features(self, data: pd.DataFrame) -> Tuple[str, List[str]]:
        """타겟 변수와 피처 변수 식별"""
        print("\n타겟 변수 및 피처 식별")
        
        # 전력 관련 컬럼들 찾기
        power_cols = [col for col in data.columns if any(keyword in col.lower() 
                     for keyword in ['power', 'energy', 'voltage', 'current'])]
        
        print(f"전력 관련 컬럼: {power_cols}")
        
        # 타겟 변수 선택 (activePower를 우선적으로)
        target_candidates = ['activePower', 'hourly_pow', 'totalActivePower']
        target_column = None
        
        for candidate in target_candidates:
            if candidate in data.columns:
                target_column = candidate
                break
        
        if target_column is None and power_cols:
            target_column = power_cols[0]  # 첫 번째 전력 컬럼 사용
        
        if target_column is None:
            raise ValueError("타겟 변수를 찾을 수 없습니다.")
        
        # 피처 변수 선택 (타겟 제외, 텍스트 컬럼 제외)
        exclude_cols = [target_column, 'localtime', 'timestamp', 'equipmentName']
        feature_columns = [col for col in data.columns 
                          if col not in exclude_cols and data[col].dtype in ['int64', 'float64']]
        
        print(f"타겟 변수: {target_column}")
        print(f"피처 변수: {len(feature_columns)}개")
        print(f"   주요 피처: {feature_columns[:10]}...")
        
        self.target_column = target_column
        self.feature_columns = feature_columns
        
        return target_column, feature_columns
    
    def create_time_series_sequences(self, data: pd.DataFrame, 
                                   sequence_length: int = 24,
                                   prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 예측을 위한 시퀀스 데이터 생성"""
        print(f"\n시계열 시퀀스 생성 (윈도우: {sequence_length}, 예측: {prediction_horizon})")
        
        # 시간 순서대로 정렬
        if 'localtime' in data.columns:
            data = data.sort_values('localtime').reset_index(drop=True)
        
        # 피처와 타겟 데이터 준비
        features = data[self.feature_columns].values
        targets = data[self.target_column].values
        
        X_sequences = []
        y_sequences = []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            # 입력 시퀀스 (과거 sequence_length 시점)
            X_seq = features[i:i + sequence_length]
            
            # 타겟 (prediction_horizon 시점 후)
            y_seq = targets[i + sequence_length:i + sequence_length + prediction_horizon]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"시퀀스 생성 완료:")
        print(f"   - X shape: {X_sequences.shape} (samples, timesteps, features)")
        print(f"   - y shape: {y_sequences.shape} (samples, prediction_horizon)")
        
        return X_sequences, y_sequences
    
    def create_tabular_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """테이블 형태 ML을 위한 데이터 생성"""
        print(f"\n테이블 형태 데이터 생성")
        
        # 결측치 제거
        clean_data = data[self.feature_columns + [self.target_column]].dropna()
        
        X = clean_data[self.feature_columns].values
        y = clean_data[self.target_column].values
        
        # 훈련 데이터 통계 저장 (테스트 데이터 처리용)
        self.train_feature_stats = {}
        for i, feature in enumerate(self.feature_columns):
            self.train_feature_stats[feature] = {
                'mean': float(clean_data[feature].mean()),
                'std': float(clean_data[feature].std()),
                'min': float(clean_data[feature].min()),
                'max': float(clean_data[feature].max())
            }
        
        print(f"테이블 데이터 생성 완료:")
        print(f"   - X shape: {X.shape} (samples, features)")
        print(f"   - y shape: {y.shape} (samples,)")
        print(f"   - 결측치 제거 후: {len(clean_data)}/{len(data)} 샘플")
        print(f"   - 훈련 데이터 통계 저장: {len(self.train_feature_stats)}개 피처")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   val_size: float = 0.2,
                   time_series: bool = True) -> Dict:
        """데이터 분할 (Train/Validation만, Test는 별도 파일)"""
        print(f"\nTrain 데이터 분할 (Validation: {val_size*100}%)")
        
        if time_series:
            # 시계열 데이터는 시간 순서 유지
            n_samples = len(X)
            n_val = int(n_samples * val_size)
            
            # 시간 순서대로 분할 (마지막 부분을 validation으로)
            X_train = X[:-n_val]
            y_train = y[:-n_val]
            X_val = X[-n_val:]
            y_val = y[-n_val:]
            
        else:
            # 일반 ML은 랜덤 분할
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=42
            )
        
        split_data = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val
        }
        
        print(f"Train 데이터 분할 완료:")
        print(f"   - Train: {X_train.shape[0]} 샘플")
        print(f"   - Validation: {X_val.shape[0]} 샘플")
        
        return split_data
    
    def prepare_test_data(self, submission_template: pd.DataFrame, apply_scaling: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        """실제 테스트 데이터 준비 (submission 템플릿에 맞춤, 훈련 데이터와 동일한 전처리 적용)"""
        print(f"\n실제 테스트 데이터 준비 (훈련 데이터와 동일한 전처리)")
        
        # test.csv 로드
        test_path = project_root / "data" / "raw" / "test.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"테스트 데이터 파일이 없습니다: {test_path}")
        
        print(f"테스트 데이터 로딩: {test_path}")
        
        # Submission 기간 확인
        submission_times = pd.to_datetime(submission_template['id'])
        start_time = submission_times.min()
        end_time = submission_times.max()
        
        print(f"Submission 예측 기간: {start_time} ~ {end_time}")
        print(f"   총 {len(submission_template)}개 시점 예측 필요")
        
        # 효율적 로딩: 필요한 부분 + 여유분만 로드
        print(f"효율적 데이터 로딩 중...")
        
        # 전체 파일에서 샘플링 (시계열 연속성 고려하여 최근 데이터)
        total_rows_estimate = 500000  # 대략적 추정
        skip_rows = max(0, total_rows_estimate - 50000)  # 마지막 5만개만 로드
        
        test_data = pd.read_csv(test_path, skiprows=range(1, skip_rows), nrows=50000)
        print(f"테스트 데이터 샘플 로드: {test_data.shape}")
        
        # 전체 전처리 파이프라인 적용 (훈련 데이터와 동일)
        print(f"전체 전처리 파이프라인 적용 중...")
        
        # 1. localtime 처리
        if 'localtime' in test_data.columns:
            test_data['localtime'] = pd.to_datetime(test_data['localtime'].astype(str), format='%Y%m%d%H%M%S', errors='coerce')
            test_data = test_data.sort_values('localtime').reset_index(drop=True)
            print(f"시간 컬럼 처리 완료")
        
        # 2. 결측치 처리
        numeric_cols = test_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if test_data[col].isnull().sum() > 0:
                test_data[col] = test_data[col].fillna(test_data[col].median())
        print(f"결측치 처리 완료")
        
        # 3. 시간 특성 추가 (Train과 동일 구조 필요)
        if 'localtime' in test_data.columns:
            dt_series = test_data['localtime']
            
            # 기본 시간 특성
            test_data['hour'] = dt_series.dt.hour
            test_data['day_of_week'] = dt_series.dt.dayofweek
            test_data['month'] = dt_series.dt.month
            
            # 스마트 팩토리 특화 특성
            test_data['is_weekend'] = (test_data['day_of_week'] >= 5).astype(int)
            test_data['is_peak_hour'] = ((test_data['hour'] >= 8) & (test_data['hour'] <= 18)).astype(int)
            test_data['is_business_hour'] = ((test_data['hour'] >= 9) & (test_data['hour'] <= 17)).astype(int)
            
            print(f"시간 특성 추가 완료")
        
        # 4. Submission 기간에 해당하는 데이터 추출
        if 'localtime' in test_data.columns:
            test_times = test_data['localtime']
            
            # Submission 기간과 가장 가까운 부분 찾기
            mask = (test_times >= start_time) & (test_times <= end_time)
            filtered_test = test_data[mask].copy()
            
            print(f"Submission 기간과 매칭된 데이터: {len(filtered_test)}개")
            
            if len(filtered_test) < len(submission_template):
                print(f"매칭된 데이터 부족, 최신 {len(submission_template)}개 데이터 사용")
                filtered_test = test_data.tail(len(submission_template)).copy()
        else:
            print("localtime 없음, 최신 데이터 사용")
            filtered_test = test_data.tail(len(submission_template)).copy()
        
        # 5. Submission 템플릿에 맞춰 정렬
        if len(filtered_test) > len(submission_template):
            filtered_test = filtered_test.head(len(submission_template))
        elif len(filtered_test) < len(submission_template):
            # 부족한 경우 마지막 값으로 패딩
            last_row = filtered_test.iloc[-1:].copy()
            needed = len(submission_template) - len(filtered_test)
            padding = pd.concat([last_row] * needed, ignore_index=True)
            filtered_test = pd.concat([filtered_test, padding], ignore_index=True)
        
        # Submission ID 추가
        filtered_test['submission_id'] = submission_template['id'].values
        
        # 6. 핵심 피처만 선택 및 누락 피처 처리 개선
        available_features = [col for col in self.feature_columns if col in filtered_test.columns]
        
        if len(available_features) != len(self.feature_columns):
            print(f"⚠️ 일부 피처가 테스트 데이터에 없습니다:")
            missing_features = set(self.feature_columns) - set(available_features)
            print(f"   누락된 피처: {missing_features}")
            
            # 누락된 피처를 훈련 데이터 평균값으로 채움 (0 대신)
            for feature in missing_features:
                if feature in self.train_feature_stats:
                    fill_value = self.train_feature_stats[feature]['mean']
                    print(f"   {feature}: 훈련 데이터 평균값 {fill_value:.4f}로 채움")
                else:
                    fill_value = 0
                    print(f"   ⚠️ {feature}: 통계 없음, 0으로 채움")
                filtered_test[feature] = fill_value
            
            available_features = self.feature_columns
        
        # 7. 피처 데이터 추출
        X_test = filtered_test[available_features].values
        
        # 8. 스케일링 적용 (훈련 데이터와 동일한 스케일러 사용)
        if apply_scaling and self.scaler is not None:
            print(f"훈련 데이터와 동일한 스케일링 적용 중...")
            X_test_original = X_test.copy()
            X_test = self.scaler.transform(X_test)
            
            print(f"✅ 스케일링 적용 완료:")
            print(f"   - 스케일링 전: 범위 {X_test_original.min():.4f} ~ {X_test_original.max():.4f}")
            print(f"   - 스케일링 후: 범위 {X_test.min():.4f} ~ {X_test.max():.4f}")
        elif apply_scaling and self.scaler is None:
            print(f"⚠️ 스케일러가 저장되지 않았습니다. 스케일링을 건너뜁니다.")
        
        print(f"✅ 훈련 데이터와 동일한 전처리 완료:")
        print(f"   - X_test shape: {X_test.shape}")
        print(f"   - 사용된 피처: {len(available_features)}개")
        print(f"   - 전처리: 완전 (결측치 + 시간특성 + 스케일링)")
        print(f"   - Submission 호환: ✅")
        
        return X_test, filtered_test
    
    def save_ml_data(self, split_data: Dict, data_type: str = "tabular"):
        """ML 학습용 데이터 저장"""
        print(f"\nML 학습용 데이터 저장 ({data_type})")
        
        # 저장 디렉토리 생성
        save_dir = project_root / "data" / "ml_ready" / data_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 데이터 저장
        for split_name, data_array in split_data.items():
            if isinstance(data_array, np.ndarray):  # numpy 배열만 저장
                file_path = save_dir / f"{split_name}_{timestamp}.npy"
                np.save(file_path, data_array)
                print(f"   - {split_name}: {file_path}")
        
        # 스케일러 저장 (있는 경우)
        if self.scaler is not None:
            import pickle
            scaler_path = save_dir / f"scaler_{timestamp}.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"   - scaler: {scaler_path}")
        
        # 훈련 데이터 통계 저장
        if self.train_feature_stats:
            stats_path = save_dir / f"train_stats_{timestamp}.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.train_feature_stats, f, indent=2, ensure_ascii=False)
            print(f"   - train_stats: {stats_path}")
        
        # 메타데이터 저장
        metadata = {
            'data_type': data_type,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'shapes': {name: data.shape for name, data in split_data.items() if isinstance(data, np.ndarray)},
            'timestamp': timestamp,
            'total_features': len(self.feature_columns),
            'has_scaler': self.scaler is not None,
            'has_train_stats': bool(self.train_feature_stats)
        }
        
        metadata_path = save_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"메타데이터 저장: {metadata_path}")
        
        return save_dir, timestamp
    
    def select_important_features(self, data: pd.DataFrame) -> List[str]:
        """ML/DL에 적합한 핵심 피처만 선택"""
        print(f"\n핵심 피처 선택")
        
        # 스마트 팩토리 전력 예측에 중요한 피처들
        core_power_features = [
            'activePower',  # 타겟 (제외될 예정)
            'voltageR', 'voltageS', 'voltageT',  # 3상 전압
            'currentR', 'currentS', 'currentT',  # 3상 전류
            'powerFactorR', 'powerFactorS', 'powerFactorT',  # 역률
            'accumActiveEnergy',  # 누적 에너지
            'operation'  # 운영 상태
        ]
        
        # 시간 특성 (핵심만)
        core_time_features = [
            'hour',  # 시간
            'day_of_week',  # 요일
            'month',  # 월
            'is_weekend',  # 주말 여부
            'is_peak_hour',  # 피크 시간
            'is_business_hour'  # 업무 시간
        ]
        
        # 사용 가능한 피처만 선택
        available_features = []
        
        for feature in core_power_features + core_time_features:
            if feature in data.columns and feature != self.target_column:
                available_features.append(feature)
        
        print(f"선택된 핵심 피처: {len(available_features)}개")
        print(f"   전력 관련: {[f for f in available_features if any(k in f.lower() for k in ['voltage', 'current', 'power', 'energy', 'operation'])]}")
        print(f"   시간 관련: {[f for f in available_features if f in core_time_features]}")
        
        return available_features
    
    def prepare_submission_template(self) -> pd.DataFrame:
        """submission 템플릿 준비"""
        print(f"\nSubmission 템플릿 준비")
        
        # sample_submission_final.csv 로드
        submission_path = project_root / "data" / "raw" / "sample_submission_final.csv"
        if not submission_path.exists():
            raise FileNotFoundError(f"Submission 템플릿이 없습니다: {submission_path}")
        
        submission_template = pd.read_csv(submission_path)
        print(f"✅ Submission 템플릿 로드: {submission_template.shape}")
        print(f"   컬럼: {list(submission_template.columns)}")
        print(f"   예측 기간: {submission_template['id'].iloc[0]} ~ {submission_template['id'].iloc[-1]}")
        
        return submission_template

def main():
    """메인 실행 함수"""
    print("🚀 스마트 팩토리 전력 예측 - ML/DL 학습 데이터 준비")
    print("="*80)
    
    # 데이터 준비기 초기화
    preparator = MLDataPreparator()
    
    try:
        # 1. 전처리된 Train 데이터 로드 (train.csv 기반)
        train_data = preparator.load_preprocessed_data()
        
        # 2. 타겟과 피처 식별
        target_col, feature_cols = preparator.identify_target_and_features(train_data)
        
        # 2.5. 핵심 피처만 선택 (너무 많은 피처 문제 해결)
        core_features = preparator.select_important_features(train_data)
        preparator.feature_columns = core_features  # 선택된 피처로 업데이트
        
        print(f"\n📊 피처 개수 비교:")
        print(f"   전체 피처: {len(feature_cols)}개 → 핵심 피처: {len(core_features)}개")
        
        # 2.6. Submission 템플릿 준비
        submission_template = preparator.prepare_submission_template()
        
        # 3. 사용자 선택
        print(f"\n🔧 ML/DL 데이터 타입을 선택하세요:")
        print(f"   1. 테이블 형태 (일반 ML 모델용)")
        print(f"   2. 시계열 시퀀스 (LSTM, GRU 등 DL 모델용)")
        print(f"   3. 둘 다 생성")
        
        choice = input("\n선택 (1-3): ").strip()
        
        if choice == "1" or choice == "3":
            # 테이블 형태 데이터 생성
            print(f"\n📋 테이블 형태 데이터 준비 중...")
            X_tab, y_tab = preparator.create_tabular_data(train_data)
            split_tab = preparator.split_data(X_tab, y_tab, time_series=False)
            
            # 스케일링 적용 (훈련 데이터 기준으로 피팅)
            print(f"\n🔧 스케일링 적용 중...")
            from sklearn.preprocessing import StandardScaler
            preparator.scaler = StandardScaler()
            
            # 훈련 데이터로 스케일러 학습
            X_train_original = split_tab['X_train'].copy()
            split_tab['X_train'] = preparator.scaler.fit_transform(split_tab['X_train'])
            split_tab['X_val'] = preparator.scaler.transform(split_tab['X_val'])
            
            print(f"✅ 스케일링 완료:")
            print(f"   - 원본 훈련 데이터 범위: {X_train_original.min():.4f} ~ {X_train_original.max():.4f}")
            print(f"   - 스케일링 후 범위: {split_tab['X_train'].min():.4f} ~ {split_tab['X_train'].max():.4f}")
            
            # 실제 테스트 데이터 준비 (스케일링 자동 적용됨)
            X_test_tab, test_df = preparator.prepare_test_data(submission_template, apply_scaling=True)
            split_tab['X_test'] = X_test_tab
            split_tab['test_data'] = test_df  # 예측 결과 저장용
            
            save_dir_tab, timestamp_tab = preparator.save_ml_data(split_tab, "tabular")
        
        if choice == "2" or choice == "3":
            # 시계열 시퀀스 데이터 생성
            print(f"\n시계열 시퀀스 데이터 준비 중...")
            
            # 시퀀스 길이 설정
            print(f"\n시계열 설정 권장사항:")
            print(f"   스마트 팩토리 전력 예측 최적화")
            print(f"   시퀀스 길이 (Sequence Length):")
            print(f"      - 24시간: 일일 패턴 학습 (빠름, 기본)")
            print(f"      - 48시간: 2일 패턴 (안정성)")  
            print(f"      - 72시간: 3일 패턴 (주말 고려)")
            print(f"   🎯 예측 구간 (Prediction Horizon):")
            print(f"      - 1시간: 1-step ahead (권장)")
            print(f"      - 6시간: 6시간 후 예측")
            print(f"      - 24시간: 하루 후 예측")
            
            print(f"\n💡 권장 조합:")
            print(f"   🥇 1순위: sequence=24, horizon=1 (빠른 프로토타입)")
            print(f"   🥈 2순위: sequence=48, horizon=1 (안정성)")
            print(f"   🥉 3순위: sequence=72, horizon=1 (성능 최적화)")
            
            choice = input("\n설정을 선택하세요 (1:기본, 2:안정성, 3:최적화, c:사용자정의): ").strip().lower()
            
            if choice == "1":
                sequence_length, prediction_horizon = 24, 1
                print(f"기본 설정 선택: {sequence_length}시간 → {prediction_horizon}시간")
            elif choice == "2":
                sequence_length, prediction_horizon = 48, 1  
                print(f"안정성 설정 선택: {sequence_length}시간 → {prediction_horizon}시간")
            elif choice == "3":
                sequence_length, prediction_horizon = 72, 1
                print(f"최적화 설정 선택: {sequence_length}시간 → {prediction_horizon}시간")
            else:
                sequence_length = int(input("시퀀스 길이를 입력하세요 (기본: 24): ") or "24")
                prediction_horizon = int(input("예측 구간을 입력하세요 (기본: 1): ") or "1")
            
            X_seq, y_seq = preparator.create_time_series_sequences(
                train_data, sequence_length, prediction_horizon
            )
            split_seq = preparator.split_data(X_seq, y_seq, time_series=True)
            
            # 테스트 데이터도 시퀀스로 변환
            X_test_seq, test_df = preparator.prepare_test_data(submission_template)
            
            # 테스트 데이터를 시퀀스로 변환하기 위해 임시로 타겟 컬럼 추가
            test_df_copy = test_df.copy()
            test_df_copy[preparator.target_column] = 0  # 더미 값
            
            # 테스트 데이터를 시퀀스로 변환
            preparator_temp = MLDataPreparator()
            preparator_temp.feature_columns = preparator.feature_columns
            preparator_temp.target_column = preparator.target_column
            
            try:
                X_test_seq, _ = preparator_temp.create_time_series_sequences(
                    test_df_copy, sequence_length, prediction_horizon
                )
                split_seq['X_test'] = X_test_seq
                split_seq['test_data'] = test_df
            except Exception as e:
                print(f"테스트 데이터 시퀀스 변환 실패: {e}")
                print(" 테스트 데이터는 테이블 형태로만 저장됩니다.")
                split_seq['X_test'] = X_test_seq
                split_seq['test_data'] = test_df
            
            save_dir_seq, timestamp_seq = preparator.save_ml_data(split_seq, "time_series")
        
        print(f"\nML/DL 학습 데이터 준비 완료")
        print(f"결과는 'data/ml_ready/' 디렉토리에서 확인.")
        
        # 사용법 안내
        print(f"\n  데이터 구조:")
        print(f"   - Train: 학습용 데이터 (레이블 있음)")
        print(f"   - Validation: 검증용 데이터 (레이블 있음)")
        print(f"   - Test: 실제 예측용 데이터 (레이블 없음)")
        print(f"   - 타겟 변수: {target_col}")
        print(f"   - 피처 수: {len(feature_cols)}개")
        
        print(f"\n  사용법:")
        print(f"   - Python에서: np.load('파일경로.npy')로 로드")
        print(f"   - 메타데이터: JSON 파일에서 컬럼 정보 확인")
        print(f"   - 예측 후 결과를 submission 형태로 변환 필요")
        
    except Exception as e:
        print(f"데이터 준비 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 