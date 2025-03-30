import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models import Encoder, Decoder, Discriminator

# -- 하이퍼파라미터 설정 --
BATCH_SIZE = 64
FEATURES = 22  # 전처리된 데이터의 feature 수
WINDOW_SIZE = 30  # 슬라이딩 윈도우 크기

def load_data(file_path):
    """전처리된 데이터 로드 및 TensorDataset 생성"""
    data = np.load(file_path)  # shape: (samples, window_size, features)
    data = torch.tensor(data, dtype=torch.float32)  # PyTorch Tensor 변환
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate(encoder, decoder, dataloader, device):
    """모델 평가 (Test)"""
    encoder.eval()  # 평가 모드로 설정
    decoder.eval()  # 평가 모드로 설정
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            latent = encoder(x)
            x_reconstructed = decoder(latent)
            loss = criterion(x_reconstructed, x)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

def load_trained_model(encoder_path, decoder_path, discriminator_path, device):
    """훈련된 모델 로드"""
    encoder = Encoder(FEATURES, 64, 32)
    decoder = Decoder(32, 64, FEATURES, WINDOW_SIZE)
    discriminator = Discriminator(FEATURES, 64)

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    discriminator.load_state_dict(torch.load(discriminator_path))

    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)

    return encoder, decoder, discriminator

if __name__ == "__main__":
    # 경로 설정 (훈련된 모델 및 테스트 데이터)
    encoder_path = "encoder.pth"
    decoder_path = "decoder.pth"
    discriminator_path = "discriminator.pth"
    test_data_path = "v_filtered.csv.npy"  # 테스트 데이터 파일 경로

    # 장치 설정 (GPU가 가능하면 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 테스트 데이터 로드
    test_dataloader = load_data(test_data_path)

    # 훈련된 모델 로드
    encoder, decoder, discriminator = load_trained_model(encoder_path, decoder_path, discriminator_path, device)

    # 모델 평가
    evaluate(encoder, decoder, test_dataloader, device)
