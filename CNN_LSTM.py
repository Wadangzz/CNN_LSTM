import numpy as np
import torch
import torch.nn as nn

class CNNLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN 계층
        self.conv1 = nn.Conv1d(input_size, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv1_bn = nn.BatchNorm1d(16)
        self.conv2_bn = nn.BatchNorm1d(32)
        self.conv3_bn = nn.BatchNorm1d(64)
        self.cnn = nn.Sequential(
            self.conv1,
            self.conv1_bn,
            self.relu,
            self.pool,
            self.conv2,
            self.conv2_bn,
            self.relu,
            self.pool,
            self.conv3,
            self.conv3_bn,
            self.relu
        )

        # LSTM 계층
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True)

        # Fully connected 계층
        self.fc1 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # CNN 계층을 통과하기 위해 입력 차원 변경
        # (batch_size, seq_len, input_size) -> (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        
        # 초기 은닉 상태와 셀 상태를 0으로 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 계층을 통과
        out, _ = self.lstm(x, (h0, c0))

        # 마지막 시퀀스의 출력을 사용하여 예측
        out = self.fc1(out[:, -1, :])

        return out