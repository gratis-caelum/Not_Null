"""
스마트 팩토리 전력 예측 - 데이터 전처리 모듈
===================================================

시계열 데이터 전처리 주요 단계:
1. 결측치(Missing Value) 처리
2. 노이즈(Noise) 제거  
3. 이상치(Outlier) 처리
4. 스케일링(Scaling)
5. 업샘플링/다운샘플링
6. 피처 엔지니어링
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union

# 프로젝트 루트 디렉토리 설정
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 한글 폰트 설정
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 8)

# 한글 폰트 설정 (운영체제별)
import platform
import matplotlib.font_manager as fm

def setup_korean_font():
    """한글 폰트 설정"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        # macOS에서 사용 가능한 한글 폰트들
        korean_fonts = ['AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic', 'NanumBarunGothic']
        
        for font in korean_fonts:
            try:
                plt.rcParams['font.family'] = font
                # 테스트해보기
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '테스트', fontsize=12)
                plt.close(fig)
                print(f"한글 폰트 설정 완료: {font}")
                break
            except:
                continue
        else:
            print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            
    elif system == 'Windows':
        try:
            plt.rcParams['font.family'] = 'Malgun Gothic'
            print("한글 폰트 설정 완료: Malgun Gothic")
        except:
            print("한글 폰트 설정 실패. 기본 폰트를 사용합니다.")
            
    else:  # Linux
        try:
            plt.rcParams['font.family'] = 'NanumGothic'
            print("한글 폰트 설정 완료: NanumGothic")
        except:
            print("한글 폰트 설정 실패. 기본 폰트를 사용합니다.")
    
    # 음수 표시 문제 해결
    plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정 실행
setup_korean_font()

class PowerDataPreprocessor:
    """
    스마트 팩토리 전력 데이터 전처리 클래스
    
    주요 기능:
    - 결측치 처리 (보간법, 시계열 분해 기반)
    - 이상치 탐지 및 처리 (IQR, 3-sigma, LOF)
    - 노이즈 필터링 (이동평균, 칼만 필터)
    - 스케일링 (Standard, MinMax, Robust)
    - 피처 엔지니어링 (시간 특성, 지연 특성, 통계적 특성)
    """
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: 전처리 설정 딕셔너리
        """
        self.config = config or self._get_default_config()
        self.scalers = {}
        self.preprocessing_history = []
        self.feature_names = []
        
    def _get_default_config(self) -> Dict:
        """기본 전처리 설정"""
        return {
            'missing_value': {
                'method': 'seasonal_decomposition',  # 'forward_fill', 'interpolate', 'seasonal_decomposition'
                'interpolation_method': 'linear',
                'seasonal_period': 24
            },
            'outlier_detection': {
                'method': 'iqr',  # 'iqr', 'zscore', 'modified_zscore', 'isolation_forest'
                'threshold': 1.5,  # IQR의 경우 1.5, z-score의 경우 3
                'action': 'cap'  # 'remove', 'cap', 'interpolate'
            },
            'noise_filtering': {
                'method': 'moving_average',  # 'moving_average', 'exponential_smoothing', 'kalman'
                'window_size': 3,
                'alpha': 0.3  # 지수평활 계수
            },
            'scaling': {
                'method': 'standard',  # 'standard', 'minmax', 'robust'
                'feature_range': (0, 1)
            },
            'feature_engineering': {
                'time_features': True,
                'lag_features': True,
                'rolling_features': True,
                'lag_periods': [1, 6, 12, 24, 48],
                'rolling_windows': [6, 12, 24]
            }
        }
    
    def preprocess_data(self, data: pd.DataFrame, target_columns: List[str] = None) -> pd.DataFrame:
        """
        데이터 전처리 파이프라인 실행
        
        Args:
            data: 원본 데이터프레임
            target_columns: 전처리할 컬럼 리스트 (None인 경우 숫자형 컬럼 모두)
            
        Returns:
            전처리된 데이터프레임
        """
        print("🔧 데이터 전처리 파이프라인 시작")
        print("="*50)
        
        # 데이터 복사
        processed_data = data.copy()
        
        # 시간 컬럼 처리
        processed_data = self._process_datetime_column(processed_data)
        
        # 타겟 컬럼 설정
        if target_columns is None:
            target_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
            # 시간 관련 컬럼 제외
            time_cols = ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'quarter']
            target_columns = [col for col in target_columns if col not in time_cols]
        
        print(f"📊 전처리 대상 컬럼: {len(target_columns)}개")
        
        # 1. 결측치 처리
        processed_data = self._handle_missing_values(processed_data, target_columns)
        
        # 2. 이상치 처리
        processed_data = self._handle_outliers(processed_data, target_columns)
        
        # 3. 노이즈 필터링
        processed_data = self._apply_noise_filtering(processed_data, target_columns)
        
        # 4. 피처 엔지니어링
        processed_data = self._feature_engineering(processed_data, target_columns)
        
        # 5. 스케일링 (마지막에 수행)
        processed_data = self._apply_scaling(processed_data, target_columns)
        
        # 처리 결과 요약
        self._print_preprocessing_summary(data, processed_data)
        
        return processed_data
    
    def _process_datetime_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """시간 컬럼 처리 및 시간 특성 추출"""
        
        # localtime 컬럼만 사용 (timestamp 무시)
        time_col = None
        if 'localtime' in data.columns:
            time_col = 'localtime'
        else:
            print("localtime 컬럼을 찾을 수 없습니다.")
            return data
        
        # localtime 컬럼 변환 (YYYYMMDDHHMMSS 형태)
        print(f"localtime 컬럼 변환 중..")
        data[time_col] = pd.to_datetime(data[time_col].astype(str), format='%Y%m%d%H%M%S', errors='coerce')
        
        # 잘못 변환된 날짜 체크
        na_count = data[time_col].isna().sum()
        if na_count > 0:
            print(f"{na_count}개의 localtime 값이 올바른 날짜로 변환되지 않았습니다.")
        
        # 시간 정렬
        data = data.sort_values(time_col).reset_index(drop=True)
        
        print(f"시간 컬럼 처리 완료: {time_col}")
        print(f"   - 시간 범위: {data[time_col].min()} ~ {data[time_col].max()}")
        print(f"   - 총 {len(data)}개 시점")
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
        """결측치 처리"""
        print("\n🔍 결측치 처리 중...")
        
        method = self.config['missing_value']['method']
        
        missing_before = data[target_columns].isnull().sum().sum()
        print(f"처리 전 결측치: {missing_before:,}개")
        
        if missing_before == 0:
            print("결측치가 없습니다.")
            return data
        
        for column in target_columns:
            if data[column].isnull().sum() > 0:
                if method == 'forward_fill':
                    data[column] = data[column].fillna(method='ffill').fillna(method='bfill')
                
                elif method == 'interpolate':
                    interp_method = self.config['missing_value']['interpolation_method']
                    data[column] = data[column].interpolate(method=interp_method)
                
                elif method == 'seasonal_decomposition':
                    data[column] = self._seasonal_decomposition_imputation(data[column])
                
                else:
                    # 기본값: 선형 보간
                    data[column] = data[column].interpolate(method='linear')
        
        missing_after = data[target_columns].isnull().sum().sum()
        print(f"처리 후 결측치: {missing_after:,}개")
        print(f"결측치 처리 완료 ({method} 방법 사용)")
        
        self.preprocessing_history.append(f"결측치 처리: {method}")
        
        return data
    
    def _seasonal_decomposition_imputation(self, series: pd.Series) -> pd.Series:
        """시계열 분해 기반 결측치 보간"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        if series.isnull().all():
            return series
        
        # 충분한 데이터가 있는 경우에만 분해 수행
        non_null_count = series.count()
        period = self.config['missing_value']['seasonal_period']
        
        if non_null_count < period * 2:
            # 데이터가 부족한 경우 선형 보간
            return series.interpolate(method='linear')
        
        try:
            # 일시적으로 결측치를 선형 보간으로 채움
            temp_series = series.interpolate(method='linear')
            
            # 시계열 분해
            decomposition = seasonal_decompose(temp_series, model='additive', period=period)
            
            # 각 성분별로 결측치 보간
            trend_filled = decomposition.trend.interpolate(method='linear')
            seasonal_filled = decomposition.seasonal.fillna(method='ffill').fillna(method='bfill')
            resid_filled = decomposition.resid.fillna(0)  # 잔차는 0으로 채움
            
            # 성분 재결합
            reconstructed = trend_filled + seasonal_filled + resid_filled
            
            # 원래 결측치 위치에만 재구성된 값 사용
            result = series.copy()
            result.loc[series.isnull()] = reconstructed.loc[series.isnull()]
            
            return result
            
        except Exception as e:
            print(f"시계열 분해 실패: {e}, 선형 보간 사용")
            return series.interpolate(method='linear')
    
    def _handle_outliers(self, data: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
        """이상치 탐지 및 처리"""
        print("\n🎯 이상치 처리 중...")
        
        method = self.config['outlier_detection']['method']
        threshold = self.config['outlier_detection']['threshold']
        action = self.config['outlier_detection']['action']
        
        total_outliers = 0
        
        for column in target_columns:
            series = data[column].dropna()
            if len(series) == 0:
                continue
                
            # 이상치 탐지
            if method == 'iqr':
                outlier_mask = self._detect_outliers_iqr(series, threshold)
            elif method == 'zscore':
                outlier_mask = self._detect_outliers_zscore(series, threshold)
            elif method == 'modified_zscore':
                outlier_mask = self._detect_outliers_modified_zscore(series, threshold)
            else:
                # 기본값: IQR
                outlier_mask = self._detect_outliers_iqr(series, threshold)
            
            outlier_count = outlier_mask.sum()
            total_outliers += outlier_count
            
            if outlier_count > 0:
                # 이상치 처리
                if action == 'remove':
                    data = data.loc[~outlier_mask]
                elif action == 'cap':
                    data.loc[outlier_mask, column] = self._cap_outliers(series, outlier_mask)
                elif action == 'interpolate':
                    data.loc[outlier_mask, column] = np.nan
                    data[column] = data[column].interpolate(method='linear')
        
        print(f"탐지된 이상치: {total_outliers:,}개")
        print(f"✅ 이상치 처리 완료 ({method}-{action} 방법 사용)")
        
        self.preprocessing_history.append(f"이상치 처리: {method}-{action}")
        
        return data
    
    def _detect_outliers_iqr(self, series: pd.Series, threshold: float = 1.5) -> pd.Series:
        """IQR 방법으로 이상치 탐지"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Z-score 방법으로 이상치 탐지"""
        z_scores = np.abs(stats.zscore(series))
        return z_scores > threshold
    
    def _detect_outliers_modified_zscore(self, series: pd.Series, threshold: float = 3.5) -> pd.Series:
        """Modified Z-score 방법으로 이상치 탐지"""
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    def _cap_outliers(self, series: pd.Series, outlier_mask: pd.Series) -> pd.Series:
        """이상치를 경계값으로 제한"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        capped_values = series.copy()
        capped_values[outlier_mask] = np.where(
            series[outlier_mask] < lower_bound, 
            lower_bound, 
            upper_bound
        )
        
        return capped_values[outlier_mask]
    
    def _apply_noise_filtering(self, data: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
        """노이즈 필터링 적용"""
        print("\n🔄 노이즈 필터링 중...")
        
        method = self.config['noise_filtering']['method']
        
        for column in target_columns:
            if method == 'moving_average':
                window_size = self.config['noise_filtering']['window_size']
                data[column] = data[column].rolling(window=window_size, center=True).mean().fillna(data[column])
            
            elif method == 'exponential_smoothing':
                alpha = self.config['noise_filtering']['alpha']
                data[column] = data[column].ewm(alpha=alpha).mean()
            
            elif method == 'kalman':
                data[column] = self._kalman_filter(data[column])
        
        print(f"✅ 노이즈 필터링 완료 ({method} 방법 사용)")
        
        self.preprocessing_history.append(f"노이즈 필터링: {method}")
        
        return data
    
    def _kalman_filter(self, series: pd.Series) -> pd.Series:
        """1차원 칼만 필터 적용"""
        # 간단한 1차원 칼만 필터 구현
        n = len(series)
        filtered = np.zeros(n)
        
        # 초기값
        x = series.iloc[0]  # 초기 상태
        P = 1.0  # 초기 공분산
        Q = 0.1  # 프로세스 노이즈
        R = 1.0  # 측정 노이즈
        
        for i in range(n):
            # 예측 단계
            x_pred = x
            P_pred = P + Q
            
            # 업데이트 단계
            if not pd.isna(series.iloc[i]):
                K = P_pred / (P_pred + R)  # 칼만 이득
                x = x_pred + K * (series.iloc[i] - x_pred)
                P = (1 - K) * P_pred
            else:
                x = x_pred
                P = P_pred
            
            filtered[i] = x
        
        return pd.Series(filtered, index=series.index)
    
    def _feature_engineering(self, data: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
        """피처 엔지니어링"""
        print("\n⚙️ 피처 엔지니어링 중...")
        
        original_columns = data.columns.tolist()
        feature_config = self.config['feature_engineering']
        
        # 시간 컬럼 찾기
        time_col = None
        for col in ['datetime', 'timestamp', 'time']:
            if col in data.columns:
                time_col = col
                break
        
        # 시간 특성 추가
        if feature_config['time_features'] and time_col:
            data = self._add_time_features(data, time_col)
        
        # 지연 특성 추가
        if feature_config['lag_features']:
            data = self._add_lag_features(data, target_columns, feature_config['lag_periods'])
        
        # 이동통계 특성 추가
        if feature_config['rolling_features']:
            data = self._add_rolling_features(data, target_columns, feature_config['rolling_windows'])
        
        # 새로 생성된 피처 이름 저장
        new_features = [col for col in data.columns if col not in original_columns]
        self.feature_names.extend(new_features)
        
        print(f"✅ 피처 엔지니어링 완료: {len(new_features)}개 피처 생성")
        
        self.preprocessing_history.append(f"피처 엔지니어링: {len(new_features)}개 피처 생성")
        
        return data
    
    def _add_time_features(self, data: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """시간 관련 피처 추가"""
        # localtime 컬럼 우선 사용
        if 'localtime' in data.columns:
            dt_series = data['localtime']
        else:
            dt_series = pd.to_datetime(data[time_col])
        
        # 기본 시간 특성
        data['hour'] = dt_series.dt.hour
        data['day_of_week'] = dt_series.dt.dayofweek
        data['day_of_month'] = dt_series.dt.day
        data['month'] = dt_series.dt.month
        data['quarter'] = dt_series.dt.quarter
        data['year'] = dt_series.dt.year
        
        # 순환 특성 (삼각함수로 인코딩)
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # 특별한 날 특성
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_business_hour'] = ((data['hour'] >= 9) & (data['hour'] <= 17)).astype(int)
        
        # 공장 운영 특성 (스마트 팩토리 특화)
        data['is_peak_hour'] = ((data['hour'] >= 8) & (data['hour'] <= 18)).astype(int)  # 주요 생산시간
        data['is_night_shift'] = ((data['hour'] >= 22) | (data['hour'] <= 6)).astype(int)  # 야간 근무
        
        print(f"시간 특성 추가 완료: {12}개 피처 생성")
        
        return data
    
    def _add_lag_features(self, data: pd.DataFrame, target_columns: List[str], lag_periods: List[int]) -> pd.DataFrame:
        """지연 특성 추가"""
        for column in target_columns:
            for lag in lag_periods:
                lag_col_name = f"{column}_lag_{lag}"
                data[lag_col_name] = data[column].shift(lag)
        
        return data
    
    def _add_rolling_features(self, data: pd.DataFrame, target_columns: List[str], windows: List[int]) -> pd.DataFrame:
        """이동통계 특성 추가"""
        for column in target_columns:
            for window in windows:
                # 이동평균
                data[f"{column}_rolling_mean_{window}"] = data[column].rolling(window=window).mean()
                
                # 이동표준편차
                data[f"{column}_rolling_std_{window}"] = data[column].rolling(window=window).std()
                
                # 이동최솟값/최댓값
                data[f"{column}_rolling_min_{window}"] = data[column].rolling(window=window).min()
                data[f"{column}_rolling_max_{window}"] = data[column].rolling(window=window).max()
                
                # 이동범위
                data[f"{column}_rolling_range_{window}"] = (
                    data[f"{column}_rolling_max_{window}"] - data[f"{column}_rolling_min_{window}"]
                )
        
        return data
    
    def _apply_scaling(self, data: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
        """스케일링 적용"""
        print("\n스케일링 중..")
        
        method = self.config['scaling']['method']
        
        # 스케일러 초기화
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            feature_range = self.config['scaling']['feature_range']
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            print(f"알 수 없는 스케일링 방법: {method}")
            return data
        
        # 스케일링 적용할 컬럼 (숫자형 컬럼만)
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        # 시간 관련 컬럼 제외
        time_related_cols = ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'quarter', 'is_weekend', 'is_business_hour']
        scaling_columns = [col for col in numeric_columns if col not in time_related_cols]
        
        if len(scaling_columns) > 0:
            # 결측치가 없는 데이터로 스케일러 학습
            clean_data = data[scaling_columns].dropna()
            if len(clean_data) > 0:
                scaler.fit(clean_data)
                
                # 전체 데이터에 스케일링 적용 (결측치 제외)
                mask = data[scaling_columns].notna().all(axis=1)
                data.loc[mask, scaling_columns] = scaler.transform(data.loc[mask, scaling_columns])
                
                # 스케일러 저장
                self.scalers[method] = scaler
                
                print(f"스케일링 완료 ({method} 방법 사용, {len(scaling_columns)}개 컬럼)")
        
        self.preprocessing_history.append(f"스케일링: {method}")
        
        return data
    
    def _print_preprocessing_summary(self, original_data: pd.DataFrame, processed_data: pd.DataFrame):
        """전처리 결과 요약 출력"""
        print("\n" + "="*50)
        print("📊 전처리 결과 요약")
        print("="*50)
        
        print(f"원본 데이터 크기: {original_data.shape}")
        print(f"처리 후 데이터 크기: {processed_data.shape}")
        print(f"새로 생성된 피처: {len(self.feature_names)}개")
        
        print(f"\n전처리 단계:")
        for i, step in enumerate(self.preprocessing_history, 1):
            print(f"  {i}. {step}")
        
        # 메모리 사용량 비교
        original_memory = original_data.memory_usage(deep=True).sum() / 1024**2
        processed_memory = processed_data.memory_usage(deep=True).sum() / 1024**2
        
        print(f"\n메모리 사용량:")
        print(f"  원본: {original_memory:.2f} MB")
        print(f"  처리 후: {processed_memory:.2f} MB")
        print(f"  증가율: {(processed_memory/original_memory-1)*100:.1f}%")
    
    def visualize_preprocessing_results(self, original_data: pd.DataFrame, 
                                      processed_data: pd.DataFrame,
                                      sample_columns: List[str] = None):
        """전처리 결과 시각화"""
        print("\n📈 전처리 결과 시각화")
        
        if sample_columns is None:
            # 숫자형 컬럼 중 처음 4개 선택
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
            sample_columns = numeric_cols[:4]
        
        fig, axes = plt.subplots(len(sample_columns), 2, figsize=(20, 5*len(sample_columns)))
        if len(sample_columns) == 1:
            axes = axes.reshape(1, -1)
        
        for i, column in enumerate(sample_columns):
            if column in original_data.columns and column in processed_data.columns:
                # 원본 데이터
                axes[i, 0].plot(original_data[column], alpha=0.7, label='원본 데이터')
                axes[i, 0].set_title(f'{column} - 원본 데이터', fontweight='bold', fontsize=14)
                axes[i, 0].set_ylabel('값', fontsize=12)
                axes[i, 0].grid(True, alpha=0.3)
                axes[i, 0].legend(fontsize=10)
                
                # 처리된 데이터
                axes[i, 1].plot(processed_data[column], alpha=0.7, label='전처리 완료', color='orange')
                axes[i, 1].set_title(f'{column} - 전처리 완료 데이터', fontweight='bold', fontsize=14)
                axes[i, 1].set_ylabel('값', fontsize=12)
                axes[i, 1].grid(True, alpha=0.3)
                axes[i, 1].legend(fontsize=10)
        
        plt.tight_layout()
        
        # reports 디렉토리 생성
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # 이미지 저장
        save_path = reports_dir / "preprocessing_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"시각화 결과 저장: {save_path}")
        
        plt.show()
    
    def save_preprocessed_data(self, data: pd.DataFrame, filename: str = None):
        """전처리된 데이터 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"preprocessed_data_{timestamp}.csv"
        
        output_path = project_root / "data" / "processed" / filename
        
        # 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장
        data.to_csv(output_path, index=False)
        
        print(f"전처리된 데이터 저장 완료: {output_path}")
        
        # 메타데이터 저장
        metadata = {
            'original_shape': data.shape,
            'preprocessing_steps': self.preprocessing_history,
            'new_features': self.feature_names,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"메타데이터 저장 완료: {metadata_path}")
        
        return output_path
    
    def get_feature_importance_data(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """피처 중요도 분석을 위한 데이터 준비"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression
        
        # 숫자형 피처만 선택
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_features:
            numeric_features.remove(target_column)
        
        # 결측치 제거
        clean_data = data[numeric_features + [target_column]].dropna()
        
        if len(clean_data) == 0:
            print("피처 중요도 분석을 위한 유효한 데이터가 없습니다.")
            return pd.DataFrame()
        
        X = clean_data[numeric_features]
        y = clean_data[target_column]
        
        # Random Forest 피처 중요도
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        # 상호정보량
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # 결과 정리
        importance_df = pd.DataFrame({
            'feature': numeric_features,
            'rf_importance': rf_importance,
            'mutual_info': mi_scores
        })
        
        # 정규화
        importance_df['rf_importance_norm'] = importance_df['rf_importance'] / importance_df['rf_importance'].max()
        importance_df['mutual_info_norm'] = importance_df['mutual_info'] / importance_df['mutual_info'].max()
        
        # 평균 중요도 계산
        importance_df['avg_importance'] = (importance_df['rf_importance_norm'] + importance_df['mutual_info_norm']) / 2
        
        return importance_df.sort_values('avg_importance', ascending=False)


class QuickPreprocessor:
    """
    빠른 전처리를 위한 경량화 클래스
    대용량 데이터에 대한 기본적인 전처리만 수행
    """
    
    @staticmethod
    def quick_clean(data: pd.DataFrame) -> pd.DataFrame:
        """기본적인 정리만 수행"""
        print("⚡ 빠른 전처리 시작...")
        
        # 1. 중복 제거
        before_dup = len(data)
        data = data.drop_duplicates()
        after_dup = len(data)
        print(f"중복 제거: {before_dup - after_dup}개 행 제거")
        
        # 2. 결측치가 50% 이상인 컬럼 제거
        threshold = 0.5
        missing_ratio = data.isnull().sum() / len(data)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
            print(f"결측치 많은 컬럼 제거: {len(cols_to_drop)}개")
        
        # 3. 상수 컬럼 제거
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        constant_cols = []
        for col in numeric_cols:
            if data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            data = data.drop(columns=constant_cols)
            print(f"상수 컬럼 제거: {len(constant_cols)}개")
        
        # 4. 기본 결측치 처리
        for col in data.select_dtypes(include=[np.number]).columns:
            data[col] = data[col].fillna(data[col].median())
        
        print(f"빠른 전처리 완료: {data.shape}")
        
        return data
    
    @staticmethod
    def add_basic_time_features(data: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """기본 시간 피처만 추가"""
        dt_series = pd.to_datetime(data[time_col])
        
        data['hour'] = dt_series.dt.hour
        data['day_of_week'] = dt_series.dt.dayofweek
        data['month'] = dt_series.dt.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        return data


if __name__ == "__main__":
    # 전처리 예제 실행
    print("🔧 전처리 모듈 테스트")
    
    # 설정 예제
    config = {
        'missing_value': {'method': 'seasonal_decomposition'},
        'outlier_detection': {'method': 'iqr', 'action': 'cap'},
        'noise_filtering': {'method': 'moving_average'},
        'scaling': {'method': 'robust'},
        'feature_engineering': {
            'time_features': True,
            'lag_features': True,
            'rolling_features': True,
            'lag_periods': [1, 6, 24],
            'rolling_windows': [6, 24]
        }
    }
    
    print("전처리 모듈 준비 완료") 