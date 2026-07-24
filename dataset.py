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
_num_to_sym = {
    1: "a", 2: "b", 3: "c", 4: "d",
    5: "e", 6: "f", 7: "g", 8: "h",
}

class ChessPlaying:
    def __init__(self, model, batch=64):
        self.model = model
        self.batch = batch

    def __call__(self):
        self.model.train()
        white_win, black_win = [], []
        i=0

        while (len(white_win) <= self.batch // 2) or (len(black_win) <= self.batch // 2):
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
                black_win += history #переполнение масивов при многократном проигрыше или выигрыше
            else:
                white_win += history

            i += 1
            print(i, len(white_win), len(black_win), outcome.winner is chess.BLACK, outcome.winner is chess.WHITE, (len(white_win) <= self.batch // 2) or (len(black_win) <= self.batch // 2))
        
        white_win = random.choices(white_win, k=self.batch // 2)
        black_win = random.choices(black_win, k=self.batch // 2)

        return white_win, black_win

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
        return torch.zeros(4, 8, dtype=torch.float64)

