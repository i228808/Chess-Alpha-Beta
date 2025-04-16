import dataclasses
from dataclasses import dataclass
import chess, chess.polyglot

@dataclass
class GameState:
    board: chess.Board = dataclasses.field(default_factory=chess.Board)
    move_history: list[chess.Move] = dataclasses.field(default_factory=list)
    zobrist: int = 0

    def __post_init__(self):
        # Use zobrist_hash() if available, else fallback to hash(fen)
        if hasattr(self.board, 'zobrist_hash'):
            self.zobrist = self.board.zobrist_hash()
        else:
            self.zobrist = hash(self.board.fen())

    # thin wrappers
    def legal_moves(self):
        return list(self.board.legal_moves)

    def push(self, move: chess.Move) -> None:
        """Push a move onto the board, update history and zobrist hash."""
        self.board.push(move)
        self.move_history.append(move)
        # Use zobrist_hash() if available, else fallback to hash(fen)
        if hasattr(self.board, 'zobrist_hash'):
            self.zobrist = self.board.zobrist_hash()
        else:
            self.zobrist = hash(self.board.fen())

    def pop(self) -> None:
        """Pop the last move, update history and zobrist hash."""
        if self.move_history:
            self.board.pop()
            self.move_history.pop()
            # Use zobrist_hash() if available, else fallback to hash(fen)
            if hasattr(self.board, 'zobrist_hash'):
                self.zobrist = self.board.zobrist_hash()
            else:
                self.zobrist = hash(self.board.fen())

    def is_terminal(self) -> bool:
        """Return True if the game is over (checkmate, stalemate, etc)."""
        return self.board.is_game_over()

    def result_str(self) -> str:
        """Return the result string (e.g., '1-0', '0-1', '1/2-1/2') if game is over, else ''"""
        return self.board.result() if self.board.is_game_over() else ''
