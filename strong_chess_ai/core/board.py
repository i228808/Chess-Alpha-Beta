import dataclasses
from dataclasses import dataclass
import chess, chess.polyglot
import numpy as np
from typing import Optional

@dataclass
class GameState:
    board: chess.Board = dataclasses.field(default_factory=chess.Board)
    move_history: list[chess.Move] = dataclasses.field(default_factory=list)
    zobrist: int = 0
    f_vec: Optional[np.ndarray] = None  # Cached feature vector for incremental evaluation

    def __post_init__(self):
        # Initialize the zobrist hash using available method or fallback to hash(fen)
        if hasattr(self.board, 'zobrist_hash'):
            self.zobrist = self.board.zobrist_hash()
        else:
            self.zobrist = hash(self.board.fen())
        # Invalidate evaluation cache
        self.invalidate_eval()

    def invalidate_eval(self) -> None:
        """
        Invalidate the cached evaluation feature vector.
        This should be called whenever the board state changes.
        """
        self.f_vec = None

    def legal_moves(self) -> list:
        """
        Return a list of legal moves for the current board state.
        
        Returns:
            list: A list of legal chess.Move objects.
        """
        return list(self.board.legal_moves)

    def push(self, move: chess.Move) -> None:
        """
        Push a move onto the board, update move history, zobrist hash, and invalidate the evaluation cache.
        
        Args:
            move (chess.Move): The move to be made.
        """
        self.board.push(move)
        self.move_history.append(move)
        # Update zobrist hash if available
        if hasattr(self.board, 'zobrist_hash'):
            self.zobrist = self.board.zobrist_hash()
        else:
            self.zobrist = hash(self.board.fen())
        self.invalidate_eval()

    def pop(self) -> None:
        """
        Pop the last move from the board, update move history, zobrist hash, and invalidate the evaluation cache.
        """
        if self.move_history:
            self.board.pop()
            self.move_history.pop()
            if hasattr(self.board, 'zobrist_hash'):
                self.zobrist = self.board.zobrist_hash()
            else:
                self.zobrist = hash(self.board.fen())
            self.invalidate_eval()

    def is_terminal(self) -> bool:
        """
        Check if the game is over (e.g., checkmate, stalemate, etc.).
        
        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.board.is_game_over()

    def result_str(self) -> str:
        """
        Return the game result as a string (e.g., '1-0', '0-1', '1/2-1/2') if the game is over.
        
        Returns:
            str: The result of the game or an empty string if not over.
        """
        return self.board.result() if self.board.is_game_over() else ''
