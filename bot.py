import torch
from torch import nn
from torch import fft


class Plankton(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(13, 2, dtype=torch.float64)
        self.brain = nn.Sequential(
            # Вход 64, +1 за правило 50 ходов,  +2 за право на рокировку, +2 случайные числа
            nn.Linear(69, 32, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, 64, bias=True, dtype=torch.float64),
            nn.ReLU(),
        )
        self.from_x = nn.Sequential(
            nn.Linear(64, 8, bias=True, dtype=torch.float64),
            nn.Softmax(-1),
        )
        self.from_y = nn.Sequential(
            nn.Linear(64, 8, bias=True, dtype=torch.float64),
            nn.Softmax(-1),
        )
        self.to_x = nn.Sequential(
            nn.Linear(64, 8, bias=True, dtype=torch.float64),
            nn.Softmax(-1),
        )
        self.to_y = nn.Sequential(
            nn.Linear(64, 8, bias=True, dtype=torch.float64),
            nn.Softmax(-1),
        )

    def forward(self, board, context):  # board size [8, 8]  context size [5]
        board = self.emb(board)  # board size [8, 8, 2]
        board = fft.fft2(board, dim=(-3, -2))  # преобразование Фурье
        board = fft.fftshift(board, dim=(-3, -2))  # сортировка частот
        board = board[2:6, 2:6, :]   # берем только самое важное из данных
        board = torch.cat([board.real.reshape(-1, 32), board.imag.reshape(-1, 32)], -1)   # выпремляем и соеденяем мнимую и действительную часть

        x = torch.cat((board[0], context), -1)
        x = self.brain(x)
        move = [
            self.from_x(x), self.from_y(x),   # UCI кодировка 
            self.to_x(x),   self.to_y(x),
        ]
        move = torch.stack(move)  # move size [4, 8]

        return move
    

class ExportPlankton(nn.Module): pass