from torch.utils.data import Dataset
import torch
import random


piece_to_channel = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q":  4, "K":  5,  # Белые
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,  # Черные
}
piece_max_num = {
    "P": 8, "N": 2, "B": 2, "R": 2, "Q": 1, "K": 1,  # Белые
    "p": 8, "n": 2, "b": 2, "r": 2, "q": 1, "k": 1,  # Черные
}

class ChessDataset(Dataset):
    def __init__(self, size, dtype = torch.float64, device = "cpu"):
        self.size = size
        self.dtype = dtype
        self.device = device

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        board = [] 
        rand_pos = self._random_position()

        for piece in ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"):
            piece_board = torch.zeros(8, 8, dtype=self.dtype, device=self.device)

            for _ in range(random.randint(0, piece_max_num[piece])):
                x, y = rand_pos.pop()
                piece_board[x, y] = 1.0

            board.append(piece_board)

        board = torch.stack(board)

        return board, board


    def _random_position(self):
        pos = [(x, y) for x in range(8) for y in range(8)]
        random.shuffle(pos)

        return pos