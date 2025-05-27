import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from CNN_LSTM import CNNLSTM

# 시퀀스 데이터를 생성하는 함수
def create_dataset(sequence, seq_len):
    data_x, data_y = [], []
    # 시퀀스 길이만큼의 데이터와 그 다음 값을 예측하는 데이터 생성
    # 예를 들어, seq_len=5일 때, [0,1,2,3,4] -> 5개의 데이터로 다음 인덱스 값 예측
    for i in range(len(sequence) - seq_len): 
        data_x.append(sequence[i:i+seq_len])
        data_y.append(sequence[i+seq_len]) 
    return np.array(data_x), np.array(data_y)

# 0~100까지의 1000개 샘플링(sin곡선)
with open('mixed_frequency_vibration.json', 'r') as f:
    data = json.load(f)

x = np.array(data['timestamp'])
y = np.array(data['value'])

# 시드 고정
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(input_size=1, hidden_size=128, num_layers=3, output_size=1).to(device)

criterion = nn.MSELoss() # 손실 함수
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # Adam optimizer 적용 학습률 0.01

seq_length = 200
X, Y = create_dataset(y, seq_length)

X = torch.FloatTensor(X).view(-1, seq_length, 1).to(device) # (batch_size, seq_len, input_size)
Y = torch.FloatTensor(Y).view(-1, 1).to(device) # (batch_size, output_size)

epochs = 300

batch_size = 64
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for epoch in range(epochs):
    model.train()
    for i, (X_batch, Y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad() # 기울기 초기화
        output = model(X_batch) # 예측값
        loss = criterion(output, Y_batch) # 손실 계산
        loss.backward() # 역전파
        optimizer.step() # 가중치 업데이트

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():   
    predicted = model(X).data.cpu().numpy() # 예측값

plt.figure(figsize=(12, 6))
plt.plot(y, label='Original Data')
plt.plot(np.arange(seq_length, len(y)), predicted, label='Predicted Data')
plt.title('LSTM Prediction')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()