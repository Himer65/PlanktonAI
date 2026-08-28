import torch
from torch import nn

class Plankton(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(13, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 3, kernel_size=3, padding=1),
            nn.LayerNorm([3, 8, 8]),
            nn.ReLU(),

            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.LayerNorm([3, 8, 8]),
            nn.ReLU(),

            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.LayerNorm([4, 8, 8]),
            nn.ReLU(),
        )
        # Вход (4 * 8 * 8), +7 контекст
        self.brain = nn.Sequential(
            nn.Linear(256 + 7, 32, bias=True),
            nn.LayerNorm(32),
            nn.ReLU(),

            nn.Linear(32, 32, bias=True),
            nn.LayerNorm(32),
            nn.ReLU(),

            nn.Linear(32, 64, bias=True),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.from_x = nn.Sequential(
            nn.Linear(64, 8, bias=True),
            nn.Softmax(dim=-1),
        )
        self.from_y = nn.Sequential(
            nn.Linear(64, 8, bias=True),
            nn.Softmax(dim=-1),
        )
        self.to_x = nn.Sequential(
            nn.Linear(64, 8, bias=True),
            nn.Softmax(dim=-1),
        )
        self.to_y = nn.Sequential(
            nn.Linear(64, 8, bias=True),
            nn.Softmax(dim=-1),
        )

    def forward(self, board, context):
        board = self.emb(board).permute(2, 0, 1)  # [2, 8, 8]
        x = self.conv(board).view(-1)  # [256]
        x = torch.cat([x, context], -1)  # [263]
        x = self.brain(x)  # [64]
        move = torch.stack([
            self.from_x(x),
            self.from_y(x),
            self.to_x(x),
            self.to_y(x)
        ])  # [4, 8]

        return move