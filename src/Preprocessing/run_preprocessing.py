"""
스마트 팩토리 전력 예측 - 데이터 전처리 실행
============================================

실제 데이터에 전처리 파이프라인을 적용하는 메인 스크립트
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 디렉토리 설정
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.Preprocessing.data_preprocessing import PowerDataPreprocessor, QuickPreprocessor

def load_data_sample(file_path: str, sample_size: int = 100000) -> pd.DataFrame:
    """
    대용량 데이터의 샘플을 로드
    
    Args:
        file_path: 데이터 파일 경로
        sample_size: 샘플 크기
        
    Returns:
        샘플 데이터프레임
    """
    print(f"📊 데이터 샘플 로딩 중... (샘플 크기: {sample_size:,})")
    
    # 파일 크기 확인
    file_size = os.path.getsize(file_path) / (1024**3)  # GB
    print(f"📁 파일 크기: {file_size:.2f} GB")
    
    # 샘플 로딩
    if file_size > 1.0:  # 1GB 이상인 경우 샘플링
        # 전체 행 수 추정
        with open(file_path, 'r') as f:
            first_line = f.readline()
            
        # 건너뛸 행 계산
        total_lines = sample_size * 10  # 대략적 추정
        skip_rows = np.random.choice(range(1, total_lines), 
                                   size=total_lines-sample_size-1, 
                                   replace=False)
        
        data = pd.read_csv(file_path, skiprows=skip_rows, nrows=sample_size)
    else:
        data = pd.read_csv(file_path)
        if len(data) > sample_size:
            data = data.sample(n=sample_size).sort_index()
    
    print(f"데이터 로딩 완료: {data.shape}")
    return data

def run_quick_preprocessing():
    """빠른 전처리 실행"""
    print("\n" + "="*80)
    print("⚡ 빠른 전처리 실행")
    print("="*80)
    
    # 데이터 로드
    data_path = project_root / "data" / "raw" / "train.csv"
    
    try:
        # 작은 샘플로 시작
        data = load_data_sample(str(data_path), sample_size=5000)
        
        print(f"\n데이터 기본 정보:")
        print(f"   - 크기: {data.shape}")
        print(f"   - 컬럼: {len(data.columns)}개")
        print(f"   - 메모리: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # 컬럼 정보 출력
        print(f"\n컬럼 정보:")
        print(f"   - 시간 컬럼: {[col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]}")
        print(f"   - 숫자형 컬럼: {len(data.select_dtypes(include=[np.number]).columns)}개")
        
        # 샘플 데이터 확인
        print(f"\n샘플 데이터:")
        print(data.head())
        
        # 빠른 전처리 적용
        cleaned_data = QuickPreprocessor.quick_clean(data.copy())
        
        # 기본 시간 특성 추가 (시간 컬럼이 있는 경우)
        time_cols = [col for col in cleaned_data.columns 
                    if any(keyword in col.lower() for keyword in ['time', 'date'])]
        
        if time_cols:
            time_col = time_cols[0]
            print(f"\n시간 특성 추가 (컬럼: {time_col})")
            cleaned_data = QuickPreprocessor.add_basic_time_features(cleaned_data, time_col)
        
        # 결과 저장
        output_path = project_root / "data" / "processed" / f"quick_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cleaned_data.to_csv(output_path, index=False)
        
        print(f"\n빠른 전처리 완료!")
        print(f"   - 원본 크기: {data.shape}")
        print(f"   - 처리 후 크기: {cleaned_data.shape}")
        print(f"   - 저장 위치: {output_path}")
        
        return cleaned_data
        
    except Exception as e:
        print(f"빠른 전처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_full_preprocessing():
    """전체 전처리 파이프라인 실행"""
    print("\n" + "="*80)
    print("🔧 전체 전처리 파이프라인 실행")
    print("="*80)
    
    # 전처리 설정
    config = {
        'missing_value': {
            'method': 'interpolate',  # 빠른 선형 보간
            'interpolation_method': 'linear'
        },
        'outlier_detection': {
            'method': 'iqr',
            'threshold': 1.5,
            'action': 'cap'
        },
        'noise_filtering': {
            'method': 'moving_average',
            'window_size': 3
        },
        'scaling': {
            'method': 'robust'  # 이상치에 강건한 스케일링
        },
        'feature_engineering': {
            'time_features': True,
            'lag_features': False,  # 처음에는 끄고 시작
            'rolling_features': False,
            'lag_periods': [1, 6, 24],
            'rolling_windows': [6, 24]
        }
    }
    
    # 전처리기 초기화
    preprocessor = PowerDataPreprocessor(config)
    
    try:
        # 데이터 로드 (10만개 샘플)
        data_path = project_root / "data" / "raw" / "train.csv"
        data = load_data_sample(str(data_path), sample_size=100000)
        
        print(f"\n전처리 전 데이터 정보:")
        print(f"   - 크기: {data.shape}")
        print(f"   - 결측치: {data.isnull().sum().sum()}개")
        print(f"   - 메모리: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # 전처리 실행
        processed_data = preprocessor.preprocess_data(data)
        
        # 결과 시각화
        sample_cols = processed_data.select_dtypes(include=[np.number]).columns[:3]
        if len(sample_cols) > 0:
            preprocessor.visualize_preprocessing_results(
                data, processed_data, list(sample_cols)
            )
        
        # 결과 저장
        output_path = preprocessor.save_preprocessed_data(processed_data)
        
        print(f"\n전체 전처리 완료!")
        print(f"   - 원본 크기: {data.shape}")
        print(f"   - 처리 후 크기: {processed_data.shape}")
        print(f"   - 새 피처: {len(preprocessor.feature_names)}개")
        print(f"   - 저장 위치: {output_path}")
        
        return processed_data
        
    except Exception as e:
        print(f"전처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_data_structure():
    """데이터 구조 분석"""
    print("\n" + "="*80)
    print("🔍 데이터 구조 분석")
    print("="*80)
    
    data_path = project_root / "data" / "raw" / "train.csv"
    
    try:
        # 헤더만 읽기
        header_data = pd.read_csv(data_path, nrows=5)
        
        print(f"기본 정보:")
        print(f"   - 컬럼 수: {len(header_data.columns)}")
        print(f"   - 파일 크기: {os.path.getsize(data_path) / (1024**3):.2f} GB")
        
        print(f"\n컬럼 목록 (처음 20개):")
        for i, col in enumerate(header_data.columns[:20]):
            dtype = header_data[col].dtype
            print(f"   {i+1:2d}. {col}: {dtype}")
        
        if len(header_data.columns) > 20:
            print(f"   ... 총 {len(header_data.columns)}개 컬럼")
        
        print(f"\n샘플 데이터:")
        print(header_data.head())
        
        # 데이터 타입 분석
        numeric_cols = header_data.select_dtypes(include=[np.number]).columns
        text_cols = header_data.select_dtypes(include=['object']).columns
        
        print(f"\n데이터 타입 분포:")
        print(f"   - 숫자형: {len(numeric_cols)}개")
        print(f"   - 텍스트형: {len(text_cols)}개")
        
        if len(text_cols) > 0:
            print(f"   - 텍스트 컬럼: {list(text_cols)}")
        
        return header_data
        
    except Exception as e:
        print(f"데이터 분석 실패: {e}")
        return None

def main():
    """메인 실행 함수"""
    print("스마트 팩토리 전력 예측 - 데이터 전처리")
    print("="*80)
    
    # 1. 데이터 구조 분석
    data_info = analyze_data_structure()
    
    if data_info is None:
        return
    
    # 2. 사용자 선택
    print(f"\n전처리 옵션을 선택하세요:")
    print(f"   1. 빠른 전처리 (기본 정리만, 빠름)")
    print(f"   2. 전체 전처리 (완전한 파이프라인, 시간 소요)")
    print(f"   3. 둘 다 실행")
    
    choice = input("\n선택 (1-3): ").strip()
    
    if choice == "1":
        result = run_quick_preprocessing()
    elif choice == "2":
        result = run_full_preprocessing()
    elif choice == "3":
        print("\n빠른 전처리부터 시작...")
        quick_result = run_quick_preprocessing()
        
        if quick_result is not None:
            print("\n전체 전처리 실행...")
            full_result = run_full_preprocessing()
    else:
        print("잘못된 선택입니다. 빠른 전처리를 실행합니다.")
        result = run_quick_preprocessing()
    
    print(f"\n전처리 작업 완료!")
    print(f"결과는 'data/processed/' 디렉토리에서 확인하세요.")

if __name__ == "__main__":
    main() 