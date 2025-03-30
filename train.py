import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse

from models import Encoder, Decoder, Discriminator

# -- 하이퍼파라미터 설정 --
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
WINDOW_SIZE = 30  # 슬라이딩 윈도우 크기

# 전처리된 데이터의 feature 수 (실제 데이터에 맞게 수정)
FEATURES = 22  

def load_data(file_path):
    """전처리된 데이터 로드 및 TensorDataset 생성"""
    data = np.load(file_path)  # shape: (samples, window_size, features)
    data = torch.tensor(data, dtype=torch.float32)  # PyTorch Tensor 변환
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def train(args):
    """MTAD-GAN 모델 학습"""
    dataloader = load_data(args.data)

    # -- 모델 초기화 --
    input_size = FEATURES   # 전처리된 데이터 feature 수
    hidden_size = 64
    latent_dim = 32

    encoder = Encoder(input_size, hidden_size, latent_dim)
    decoder = Decoder(latent_dim, hidden_size, input_size, WINDOW_SIZE)
    discriminator = Discriminator(input_size, hidden_size)

    # -- GPU 사용 가능 여부 확인 --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)

    # -- 옵티마이저 및 손실 함수 설정 --
    optimizer_G = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()  # 재구성 손실로 MSE 사용

    # -- 학습 루프 --
    for epoch in range(EPOCHS):
        for batch in dataloader:
            x = batch[0].to(device)  # x: (batch_size, window_size, input_size)

            # Generator (Encoder-Decoder) 학습: 재구성 손실 최소화
            optimizer_G.zero_grad()
            latent = encoder(x)
            x_reconstructed = decoder(latent)
            recon_loss = criterion(x_reconstructed, x)
            recon_loss.backward()
            optimizer_G.step()

            # Discriminator 학습
            optimizer_D.zero_grad()
            real_out = discriminator(x)
            fake_out = discriminator(x_reconstructed.detach())
            d_loss = (criterion(real_out, torch.ones_like(real_out)) +
                      criterion(fake_out, torch.zeros_like(fake_out)))
            d_loss.backward()
            optimizer_D.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | Recon Loss: {recon_loss.item():.4f} | D Loss: {d_loss.item():.4f}")

    # -- 모델 저장 --
    torch.save(encoder.state_dict(), args.encoder_path)
    torch.save(decoder.state_dict(), args.decoder_path)
    torch.save(discriminator.state_dict(), args.discriminator_path)
    print(f"Encoder saved to {args.encoder_path}")
    print(f"Decoder saved to {args.decoder_path}")
    print(f"Discriminator saved to {args.discriminator_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="t_filtered.csv.npy", help=r"C:\Users\leedowon\OneDrive\바탕 화면\mtad-gan\edd_parameter")
    parser.add_argument("--encoder_path", type=str, default="encoder.pth", help=r"C:\Users\leedowon\OneDrive\바탕 화면\mtad-gan\edd_parameter")
    parser.add_argument("--decoder_path", type=str, default="decoder.pth", help=r"C:\Users\leedowon\OneDrive\바탕 화면\mtad-gan\edd_parameter")
    parser.add_argument("--discriminator_path", type=str, default="discriminator.pth", help=r"C:\Users\leedowon\OneDrive\바탕 화면\mtad-gan\edd_parameter")
    args = parser.parse_args()
    
    train(args)
