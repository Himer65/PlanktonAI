import torch
from torch import nn

#квантизация прунинг
class Plankton(nn.Module):
    def __init__(self,
        num_conv_layer,
        conv_hidden_chanel,
        num_brain_layer,
        brain_hidden_dim,
    ):
        super().__init__()
        import copy
        self.emb = nn.Embedding(13, 2)

        conv_layer = nn.Sequential(
            nn.Conv2d(conv_hidden_chanel, conv_hidden_chanel, kernel_size=3, padding=1),
            nn.LayerNorm([conv_hidden_chanel, 8, 8]),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(2, conv_hidden_chanel, kernel_size=3, padding=1),
            nn.LayerNorm([conv_hidden_chanel, 8, 8]),
            nn.ReLU(),

            *[copy.deepcopy(conv_layer) for _ in range(num_conv_layer)],

            nn.Conv2d(conv_hidden_chanel, conv_hidden_chanel, kernel_size=3),
            nn.LayerNorm([conv_hidden_chanel, 6, 6]),
            nn.ReLU(),

            nn.Conv2d(conv_hidden_chanel, conv_hidden_chanel, kernel_size=3),
            nn.LayerNorm([conv_hidden_chanel, 4, 4]),
            nn.ReLU(),
        )

        brain_layer = nn.Sequential(
            nn.Linear(brain_hidden_dim, brain_hidden_dim),
            nn.LayerNorm([brain_hidden_dim]),
            nn.ReLU(),
        )
        self.brain = nn.Sequential(
            nn.Linear(conv_hidden_chanel * 16 + 7, brain_hidden_dim),
            nn.LayerNorm(brain_hidden_dim),
            nn.ReLU(),

            *[copy.deepcopy(brain_layer) for _ in range(num_brain_layer)]
        )

        self.from_x = nn.Sequential(
            nn.Linear(brain_hidden_dim, 8),
            nn.Softmax(dim=-1),
        )
        self.from_y = nn.Sequential(
            nn.Linear(brain_hidden_dim, 8),
            nn.Softmax(dim=-1),
        )
        self.to_x = nn.Sequential(
            nn.Linear(brain_hidden_dim, 8),
            nn.Softmax(dim=-1),
        )
        self.to_y = nn.Sequential(
            nn.Linear(brain_hidden_dim, 8),
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