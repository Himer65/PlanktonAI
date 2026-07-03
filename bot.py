import torch
from torch import nn


class EncoderBoard(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 8, 1, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.Conv2d(8, 8, 2, bias=True, dtype=torch.float64),
            nn.Conv2d(8, 8, 2, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.Conv2d(8, 8, 2, bias=True, dtype=torch.float64),
            nn.Conv2d(8, 8, 2, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.Conv2d(8, 4, 1, bias=True, dtype=torch.float64),
            nn.Softsign(),
        )

    def forward(self, board):
        x = self.conv(board)

        return x


class DecoderBoard(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 1, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 2, bias=True, dtype=torch.float64),
            nn.ConvTranspose2d(8, 8, 2, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 2, bias=True, dtype=torch.float64),
            nn.ConvTranspose2d(8, 8, 2, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 12, 1, bias=True, dtype=torch.float64),
            nn.Sigmoid(),
        )

    def forward(self, board):
        x = self.conv(board)

        return x


class Plankton(nn.Module):
    def __init__(self, enc: EncoderBoard):
        super().__init__()
        enc.eval()
        self.enc = enc
        self.fc = nn.Sequential(
            nn.Linear(69, 32, bias=True, dtype=torch.float64),  # Вход 64, +1 за правило 50 ходов,  +2 право на рокировку, +2 случайные числа
            nn.ReLU(),
            nn.Linear(32, 32, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True, dtype=torch.float64),
            nn.Unflatten(-1, (4, 8)),
            nn.LogSoftmax(-1),
        )

    def forward(self, board, context):
        with torch.no_grad():
            x = self.enc(board)
            x = torch.flatten(x, start_dim=1)

        x = torch.cat((x, context), -1)
        x = self.fc(x)

        return x
    
    def train(self, mode = True):
        return self.fc.train(mode)
    
    def parameters(self, recurse = True):
        return self.fc.parameters(recurse)