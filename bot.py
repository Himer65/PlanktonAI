import torch
from torch import nn

class Plankton(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(13, 2, dtype=torch.float64)
        # Вход (8 * 8 * 2), +1 за правило 50 ходов,  +2 за право на рокировку
        self.brain = nn.Sequential(
            nn.Linear(128 + 3, 32, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, 64, bias=True, dtype=torch.float64),
            nn.ReLU(),
        )
        self.from_x = nn.Sequential(
            nn.Linear(64, 8, bias=True, dtype=torch.float64),
            nn.Softmax(dim=-1),
        )
        self.from_y = nn.Sequential(
            nn.Linear(64, 8, bias=True, dtype=torch.float64),
            nn.Softmax(dim=-1),
        )
        self.to_x = nn.Sequential(
            nn.Linear(64, 8, bias=True, dtype=torch.float64),
            nn.Softmax(dim=-1),
        )
        self.to_y = nn.Sequential(
            nn.Linear(64, 8, bias=True, dtype=torch.float64),
            nn.Softmax(dim=-1),
        )

    def forward(self, board, context):
        board = self.emb(board).view(-1)  # [128]
        x = torch.cat((board, context), dim=-1)  # [131]
        x = self.brain(x)  # [32]
        move = torch.stack([
            self.from_x(x),
            self.from_y(x),
            self.to_x(x),
            self.to_y(x)
        ])  # [4, 8]

        return move