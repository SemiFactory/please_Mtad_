import numpy as np

# .npy 파일 불러오기
data = np.load(r"C:\Users\leedowon\OneDrive\바탕 화면\mtad-gan\t_filtered.csv.npy")

# 배열의 크기 확인 (행렬의 형태 확인)
print("Data shape:", data.shape)

# 첫 몇 개의 데이터를 출력하여 어떤 값들이 들어있는지 확인
# 첫 번째 샘플만 확인 (첫 번째 행)
print(data[0])  # 첫 번째 샘플 (window_size, features)

