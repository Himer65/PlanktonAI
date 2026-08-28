import chess
import copy
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

class SelfPlayENV:
    def __init__(self, model, num_self_play=50):
        self.trainee = model
        self.enemy = copy.deepcopy(model)
        self.num_self_play = num_self_play
        self._num_games = 0

    def __call__(self):
        while True:
            color = random.choice([chess.WHITE, chess.BLACK])
            moves, indices, outcome = self.go_to_game(color)

            if outcome is None: continue

            self._num_games = self._num_games  + 1
            if self._num_games == self.num_self_play:
                self.enemy = copy.deepcopy(self.trainee)
                self.enemy.eval()
                self._num_games = 0

            if outcome.winner is color:
                return moves, indices, -1
                        
            elif outcome.winner is not None:
                return moves, indices,  1

            else:
                return moves, indices,  0

    def go_to_game(self, color):
        board = chess.Board()
        self.trainee.train()
        self.enemy.eval()
        history_moves = []
        history_indices = []

        while not board.is_game_over():
            if board.is_seventyfive_moves(): break                
            elif board.fullmove_number >= 200: break
        
            if board.turn == color:
                _ = self.move_model(self.enemy, board)
        
            else:
                probs, move_idx = self.move_model(self.trainee, board)

                history_moves.append(probs)
                history_indices.append(move_idx)


        return history_moves, history_indices, board.outcome() 

    def move_model(self, model, board):
        board_ten = self.board_to_tensor(board.board_fen())
        ctx = self.board_to_ctx(board)
        out = model(board_ten, ctx)
        
        probs, legal_moves = self.get_action_probs(board, out)
        
        move_idx = torch.multinomial(probs, 1).item()
        move = legal_moves[move_idx]
        board.push(move)
        
        return probs, move_idx
    
    def board_to_tensor(self, fen):
        fen = fen.replace("/", "")
        board = []
        for x in fen:
            board += _piece[x]
        board = torch.tensor(board, dtype=torch.long)

        return board.view(8, 8)

    def board_to_ctx(self, board):
        # правило 50 ходов
        fifty_moves = board.halfmove_clock / 50.0

        # права рокировки для белых и чёрных
        bk = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        bq = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        wk = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        wq = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0

        # очерёдность хода
        turn = 1.0 if board.turn == chess.BLACK else -1.0

        # взятие на проходе (клетка назначения, если нет 0)
        ep = board.ep_square
        ep = ep / 63.0 if ep is not None else 0.0

        # размер [7]
        ctx = torch.tensor([fifty_moves, bk, bq, wk, wq, turn, ep])
        return ctx
    
    def get_move_logits(self, out, move):
        from_x = _sym_to_idx[chess.square_name(move.from_square)[0]]
        from_y = chess.square_rank(move.from_square)
        to_x   = _sym_to_idx[chess.square_name(move.to_square)[0]]
        to_y   = chess.square_rank(move.to_square)
    
        log_p = (out[0, from_x].log() +
                 out[1, from_y].log() +
                 out[2, to_x].log() +
                 out[3, to_y].log())
        return log_p

    def get_action_probs(self, board, out):
        legal_moves = list(board.legal_moves)

        if not legal_moves: return torch.tensor([]), []

        logits = torch.stack([self.get_move_logits(out, m) for m in legal_moves])
        probs = torch.softmax(logits, dim=-1)

        return probs, legal_moves