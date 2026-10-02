import torch
from torch.utils.data import Dataset


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

def board_to_tensor(fen):
    fen = fen.split(" ")[0]
    fen = fen.replace("/", "")
    board = []
    for x in fen:
        board += _piece[x]
    board = torch.tensor(board, dtype=torch.long)

    return board.view(8, 8)

def board_to_context(fen): pass

def logmove_to_uci(logits, legal_moves): pass

class ChessData(Dataset): pass