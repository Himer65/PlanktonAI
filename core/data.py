import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import pandas as pd


_piece = {
    "P": [1], "N": [2], "B": [3], "R": [4],  "Q": [5],  "K": [6],
    "p": [7], "n": [8], "b": [9], "r": [10], "q": [11], "k": [12],
    "1": [0],
    "2": [0, 0],
    "3": [0, 0, 0],
    "4": [0, 0, 0, 0],
    "5": [0, 0, 0, 0, 0],
    "6": [0, 0, 0, 0, 0, 0],
    "7": [0, 0, 0, 0, 0, 0, 0],
    "8": [0, 0, 0, 0, 0, 0, 0, 0],
}
_sym_to_idx = {
    "a": 0, "b": 1, "c": 2, "d": 3,
    "e": 4, "f": 5, "g": 6, "h": 7,
}
_castling = {
    "K": torch.tensor([1., 0., 0., 0.]),
    "Q": torch.tensor([0., 1., 0., 0.]),
    "k": torch.tensor([0., 0., 1., 0.]),
    "q": torch.tensor([0., 0., 0., 1.]),
}

def _castling_rights(rights):
    if rights == "-":
        return torch.tensor([0., 0., 0., 0.])

    out = sum(map(lambda x: _castling[x], list(rights)))
    return out

def board_to_tensor(fen):
    fen = fen.split(" ")[0]
    fen = fen.replace("/", "")
    board = []
    for x in fen:
        board += _piece[x]
    board = torch.tensor(board, dtype=torch.long)

    return board.view(8, 8)

def board_to_context(fen):
    fen = fen.split(" ")[1:]
    color = torch.tensor([1., 0.]) if fen[0] == "w" else torch.tensor([0., 1.])
    rights = _castling_rights(fen[1])
    halfmove = torch.tensor([int(fen[3]) / 50])

    ctx = torch.cat([color, rights, halfmove])
    return ctx

def move_to_tensor(uci):
    uci = torch.tensor([
        _sym_to_idx[uci[0]],
        int(uci[1]) - 1, 
        _sym_to_idx[uci[2]],
        int(uci[3]) - 1,
    ])

    return F.one_hot(uci, num_classes=8).float()

class ChessData(Dataset):
    def __init__(self, path_data):
        table = pd.read_csv(path_data, usecols=["FEN", "Move"])
        table.dropna(inplace=True)
        fen  = table["FEN"]
        move = table["Move"]

        self.board = fen.apply(lambda x: board_to_tensor(x))
        self.ctx   = fen.apply(lambda x: board_to_context(x))
        self.move  = move.apply(lambda x: move_to_tensor(x))

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        board = self.board[idx]
        ctx   = self.ctx[idx]
        move  = self.move[idx]

        return board, ctx, move
