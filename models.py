import torch
import torch.nn as nn

# Encoder: 입력 시계열 데이터를 잠재 공간(latent space)으로 인코딩
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_dim)
    
    def forward(self, x):
        # x: (batch, window_size, input_size)
        _, (h_n, _) = self.lstm(x)
        latent = self.fc(h_n[-1])
        return latent

# Decoder (Generator): 잠재 공간에서 시계열 데이터를 복원
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, output_size, window_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.window_size = window_size

    def forward(self, latent):
        # latent: (batch, latent_dim)
        hidden = self.fc(latent)
        # hidden state를 window_size만큼 반복해서 LSTM 입력 생성
        hidden_seq = hidden.unsqueeze(1).repeat(1, self.window_size, 1)
        out, _ = self.lstm(hidden_seq)
        reconstructed = self.output_layer(out)
        return reconstructed

# Discriminator: 진짜 시계열 데이터와 생성된 데이터를 구분
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out
