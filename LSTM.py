import numpy as np
import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional = False)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # 초기 은닉 상태와 셀 상태를 0으로 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 계층을 통과
        out, _ = self.lstm(x, (h0, c0))

        # 마지막 시퀀스의 출력을 사용하여 예측
        out = self.fc(out[:, -1, :])
        return out
    
class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=1.0, high_weight=3.0):
        super().__init__()
        self.threshold = threshold
        self.high_weight = high_weight

    def forward(self, pred, target):
        weights = torch.where(target > self.threshold, self.high_weight, 1.0)
        loss = weights * (pred - target) ** 2
        return loss.mean()