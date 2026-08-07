import chess
import random
import torch


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

class ChessPlaying:
    def __init__(self, model, batch=32):
        self.model = model
        self.batch = batch

    def __call__(self):
        self.model.train()
        white_win, black_win = [], []

        while (len(white_win) < self.batch) or (len(black_win) < self.batch):
            board = chess.Board()
            history = []
            
            while not board.is_game_over():
                if board.is_seventyfive_moves(): break
                elif board.fullmove_number >= 200: break

                if board.turn == chess.WHITE:
                    move = random.choice(list(board.legal_moves))
                    board.push(move)

                else:
                    board_ten = self._board_to_tensor(board.board_fen())
                    ctx = self._board_to_ctx(board)
                    out = self.model(board_ten, ctx)
                    history.append(out)
                    move = self._ten_to_move(out, list(board.legal_moves))
                    board.push(move)

            outcome = board.outcome()
            
            if outcome is None:
                continue
            elif outcome.winner is chess.BLACK:
                black_win += history
                black_win = black_win[-self.batch:]
            else:
                white_win += history
                white_win = white_win[-self.batch:]
        
        white_win = random.choices(white_win, k=self.batch)
        black_win = random.choices(black_win, k=self.batch)

        return torch.stack(white_win), torch.stack(black_win)

    def _board_to_tensor(self, fen):
        fen = fen.replace("/", "")
        board = []
        for x in fen:
            board += _piece[x]
        board = torch.tensor(board, dtype=torch.long)

        return board.view(8, 8)

    def _board_to_ctx(self, board):
        fifty_moves = board.halfmove_clock / 150.0  # правило 75 ходов
        kingside = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0  # рокировка в короткую сторону
        queenside = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0  # рокировка в длинную сторону
        ctx = torch.tensor([fifty_moves, kingside, queenside,
                            random.random(), random.random()],
                            dtype=torch.float64)
        
        return ctx
    
    @torch.no_grad()
    def _ten_to_move(self, ten, legal_moves):
        L = 1000
        max_move = None

        for move in legal_moves:
            move_ten = self._move_to_ten(move)
            l = ((ten - move_ten) ** 2).mean()
            if L > l: 
                L = l
                max_move = move

        return max_move

    def _move_to_ten(self, move):
        move = str(move)
        ten = torch.zeros(4, 8, dtype=torch.float64)
        ten[0, _sym_to_idx[move[0]]] = 1.0
        ten[1,     int(move[1]) - 1] = 1.0
        ten[2, _sym_to_idx[move[2]]] = 1.0
        ten[3,     int(move[3]) - 1] = 1.0

        return ten