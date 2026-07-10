import torch
from torch import nn


class Plankton(nn.Module):
    def __init__(self):
        super().__init__()
        self.brain = nn.Sequential(
            nn.Linear(69, 32, bias=True, dtype=torch.float64),  # Вход 64, +1 за правило 50 ходов,  +2 за право на рокировку, +2 случайные числа
            nn.ReLU(),
            nn.Linear(32, 32, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, 64, bias=True, dtype=torch.float64),
            nn.ReLU(),
        )
        self.from_x = nn.Linear(64, 8, bias=True, dtype=torch.float64)#logsoftmax
        self.from_y = nn.Linear(64, 8, bias=True, dtype=torch.float64)
        self.to_x = nn.Linear(64, 8, bias=True, dtype=torch.float64)
        self.to_y = nn.Linear(64, 8, bias=True, dtype=torch.float64)

    def forward(self, board, context):
        x = torch.cat((board, context), -1)
        x = self.brain(x)

        return x