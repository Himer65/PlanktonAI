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
    def __init__(self, model):
        self.model = model

    def __call__(self):
        while True:
            moves, outcome = self.go_to_game()

            if outcome is None:
                continue

            elif outcome.winner is chess.BLACK:
                return moves,  1
                        
            elif outcome.winner is chess.WHITE:
                return moves, -1

            else:
                return moves, 0

    def go_to_game(self):
        self.model.train()
        board = chess.Board()
        history_moves = []

        while not board.is_game_over():
            if board.is_seventyfive_moves(): break                
            elif board.fullmove_number >= 200: break
        
            if board.turn == chess.WHITE:
                move = random.choice(list(board.legal_moves))
                board.push(move)
        
            else:
                board_ten = self.board_to_tensor(board.board_fen())
                ctx = self.board_to_ctx(board)
                out = self.model(board_ten, ctx)
                history_moves.append(out)
        
                move = self.ten_to_move(out, list(board.legal_moves))
                board.push(move)

        return torch.stack(history_moves), board.outcome() 
    
    def board_to_tensor(self, fen):
        fen = fen.replace("/", "")
        board = []
        for x in fen:
            board += _piece[x]
        board = torch.tensor(board, dtype=torch.long)

        return board.view(8, 8)

    def board_to_ctx(self, board):
        fifty_moves = board.halfmove_clock / 150.0  # правило 75 ходов
        kingside = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0  # рокировка в короткую сторону
        queenside = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0  # рокировка в длинную сторону
        ctx = torch.tensor([fifty_moves, kingside, queenside,
                            random.random(), random.random()],
                            dtype=torch.float64)
        
        return ctx
    
    @torch.no_grad()
    def ten_to_move(self, ten, legal_moves):
        L = 1000
        max_move = None

        for move in legal_moves:
            move_ten = self.move_to_ten(move)
            l = ((ten - move_ten) ** 2).mean()
            if L > l: 
                L = l
                max_move = move

        return max_move

    def move_to_ten(self, move):
        move = str(move)
        ten = torch.zeros(4, 8, dtype=torch.float64)
        ten[0, _sym_to_idx[move[0]]] = 1.0
        ten[1,     int(move[1]) - 1] = 1.0
        ten[2, _sym_to_idx[move[2]]] = 1.0
        ten[3,     int(move[3]) - 1] = 1.0

        return ten