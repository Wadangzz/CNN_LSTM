import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from CNN_LSTM import CNNLSTM, fft_loss

def plot_fft(signal1, signal2):
    n = min(len(signal1), len(signal2))
    fft1 = np.abs(np.fft.fft(signal1[:n]))
    fft2 = np.abs(np.fft.fft(signal2[:n]))
    freq = np.fft.fftfreq(n)

    plt.figure(figsize=(12, 6))
    plt.plot(freq[:n//2], fft1[:n//2], label="Original FFT")
    plt.plot(freq[:n//2], fft2[:n//2], label="Predicted FFT")
    plt.title("FFT Spectrum Comparison")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

# 시퀀스 데이터를 생성하는 함수
def create_dataset(sequence, seq_len, pred_len):
    data_x, data_y = [], []
    # 시퀀스 길이만큼의 데이터와 그 다음 값을 예측하는 데이터 생성
    # 예를 들어, seq_len=5일 때, [0,1,2,3,4] -> 5개의 데이터로 다음 인덱스 값 예측
    for i in range(len(sequence) - seq_len - pred_len): 
        data_x.append(sequence[i:i+seq_len])
        data_y.append(sequence[i+seq_len:i+seq_len+pred_len]) 
    return np.array(data_x), np.array(data_y)

seq_length = 200
pred_length = 50

with open('mixed_frequency_vibration.json', 'r') as f:
    data = json.load(f)

x = np.array(data['timestamp'])
y = np.array(data['value'])



# 시드 고정
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(input_size=1, hidden_size=128, num_layers=3, output_size=pred_length).to(device)

criterion = nn.MSELoss() # 손실 함수
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # Adam optimizer 적용 학습률 0.01


X, Y = create_dataset(y, seq_length, pred_length)

X = torch.FloatTensor(X).view(-1, seq_length, 1).to(device) # (batch_size, seq_len, input_size)
Y = torch.FloatTensor(Y).to(device) # (batch_size, output_size)

epochs = 200
batch_size = 64

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for epoch in range(epochs):
    model.train()
    for X_batch, Y_batch in dataloader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        mse = criterion(output, Y_batch)
        freq = fft_loss(output, Y_batch)
        loss = mse + 0.2 * freq
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()

with torch.no_grad():   
    predicted = model(X).data.cpu().numpy() # 예측값

# 결과 스티칭
stitched = np.zeros(len(y))
count = np.zeros(len(y))

for i in range(len(predicted)):
    start = seq_length + i
    end = start + pred_length
    if end > len(y):
        break
    stitched[start:end] += predicted[i][:min(pred_length, len(y) - start)]
    count[start:end] += 1
stitched = stitched / np.maximum(count, 1)


# 시각화
plt.figure(figsize=(12, 6))
plt.plot(y, label='Original Data')
plt.plot(stitched, label='Stitched Prediction')
plt.title('CNN-LSTM Multi-step Prediction')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

plot_fft(y, stitched)