import torch
from torch import nn
#квантизация прунинг
class Plankton(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(13, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 3, kernel_size=3, padding=1),
            nn.LayerNorm([3, 8, 8]),
            nn.ReLU(),

            nn.Conv2d(3, 4, kernel_size=3),
            nn.LayerNorm([4, 6, 6]),
            nn.ReLU(),
        )
        # Вход (4 * 6 * 6), +7 контекст
        self.brain = nn.Sequential(
            nn.Linear(144 + 7, 64),
            nn.LayerNorm(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        self.from_x = nn.Sequential(
            nn.Linear(32, 8),
            nn.Softmax(dim=-1),
        )
        self.from_y = nn.Sequential(
            nn.Linear(32, 8),
            nn.Softmax(dim=-1),
        )
        self.to_x = nn.Sequential(
            nn.Linear(32, 8),
            nn.Softmax(dim=-1),
        )
        self.to_y = nn.Sequential(
            nn.Linear(32, 8),
            nn.Softmax(dim=-1),
        )

    def forward(self, board, context, move_hist=None):
        board = self.emb(board).permute(2, 0, 1)  # [2, 8, 8]
        x = self.conv(board).view(-1)  # [151]
        x = torch.cat([x, context], -1)  # [151]
        x = self.brain(x)  # [32]
        move = torch.stack([
            self.from_x(x),
            self.from_y(x),
            self.to_x(x),
            self.to_y(x)
        ])  # [4, 8]

        return move