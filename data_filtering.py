import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse

def load_data(file_path):
    """CSV 파일을 로드하고 기본적인 전처리를 수행"""
    df = pd.read_csv(file_path)
    # 시간 순서 정렬 (year, month, weekday, hour, minute, second 포함)
    df = df.sort_values(by=["year", "month", "weekday", "hour", "minute", "second"]).reset_index(drop=True)
    return df

def preprocess_data(df):
    """필요한 특징을 선택하고 전처리 수행"""
    # 제거할 컬럼: device_name, year, month
    df = df.drop(columns=["device_name", "year", "month"], errors='ignore')
    
    # 시간 변수 변환 (주기성 반영)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["second_sin"] = np.sin(2 * np.pi * df["second"] / 60)
    df["second_cos"] = np.cos(2 * np.pi * df["second"] / 60)
    
    # 시간 원본 변수 삭제 (weekday, is_weekend는 그대로 유지)
    df = df.drop(columns=["hour", "minute", "second"], errors='ignore')
    
    # 데이터 정규화 (모든 수치형 변수에 대해)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
    
    return df_scaled, scaler

def create_time_windows(data, window_size=30):
    """슬라이딩 윈도우를 생성하여 시계열 데이터 학습 준비"""
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data.iloc[i:i + window_size].values)
    return np.array(sequences)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help=r"C:\Users\leedowon\OneDrive\바탕 화면\mtad-gan\training_data\validation2_data.csv")
    parser.add_argument("--output", type=str, required=True, help=r"C:\Users\leedowon\OneDrive\바탕 화면\mtad-gan\v_filtered.csv")

    args = parser.parse_args()                                      #파일 열어놓지마마
    
    df = load_data(args.input)
    df_scaled, scaler = preprocess_data(df)
    time_series_data = create_time_windows(df_scaled, window_size=30)
    
    np.save(args.output, time_series_data)
    print("Processed Data Shape:", time_series_data.shape)
    print(f"Processed data saved to {args.output}")
