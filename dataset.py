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

class ChessPlaying:
    def __init__(self, model, batch=64):
        self.model = model
        self.batch = batch

    def __call__(self):
        self.model.train()
        white_win, black_win = [], []

        while (len(white_win) >= self.batch // 2) and (len(black_win) >= self.batch // 2):
            board = chess.Board()
            history = []

            while not board.is_game_over():
                if board.is_seventyfive_moves(): break
                elif board.fullmove_number() >= 200: break

                if board.turn == chess.WHITE:
                    move = random.choice(board.legal_moves)
                    board.push(move)

                else:
                    board_ten = self._board_to_tensor(board.board_fen())
                    ctx = self._board_to_ctx(board)
                    out = self.model(board_ten, ctx)
                    history.append(out)
                    move = self._ten_to_uci(out)
                    board.push(move)

            outcome = board.outcome()
            if outcome is None:
                continue
            elif outcome.winner == chess.BLACK:
                black_win += history #переполнение масивов при многократном проигрыше или выигрыше
            else:
                white_win += history

        white_win = random.choices(white_win, k=self.batch // 2)
        black_win = random.choices(black_win, k=self.batch // 2)

        return white_win, black_win

    def _board_to_tensor(self, fen):
        fen = fen.replace("/", "")
        board = [*_piece[x] for x in fen]
        board = torch.tensor(board, dtype=torch.long)

        return board.view(8, 8)

    def _board_to_ctx(self, board):
        fifty_moves = board.halfmove_clock / 150.0  # правило 75 ходов
        kingside = 1.0 if board.has_kingside_castling_rights() else 0.0  # рокировка в короткую сторону
        queenside = 1.0 if board.has_queenside_castling_rights() else 0.0  # рокировка в длинную сторону
        ctx = torch.tensor([fifty_moves, kingside, queenside,
                            random.random(), random.random()],
                            dtype=torch.float64)
        
        return ctx
    
    def _ten_to_uci(self, ten): pass
