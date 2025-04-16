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
    f_vec: Optional[np.ndarray] = None  # Name: Abdullah Mansoor, Roll Number: i228808
# Stores the cached evaluation vector for the current board position.

    def __post_init__(self):
        # Set up the Zobrist hash for the board, using the board's method if available.
        if hasattr(self.board, 'zobrist_hash'):
            self.zobrist = self.board.zobrist_hash()
        else:
            self.zobrist = hash(self.board.fen())
        # Reset the cached evaluation data to ensure it's up-to-date.
        self.invalidate_eval()

    def invalidate_eval(self) -> None:
        """
        Resets the cached evaluation vector to None, indicating that it needs to be recalculated.
        """
        self.f_vec = None

    def legal_moves(self) -> list:
        """
        Returns a list of all legal moves for the current board state.
        """
        return list(self.board.legal_moves)

    def push(self, move: chess.Move) -> None:
        """
        Makes a move on the board, updates the move history and Zobrist hash, and resets the cached evaluation data.
        
        Args:
            move (chess.Move): The move to be made.
        """
        self.board.push(move)
        self.move_history.append(move)
        # Update the Zobrist hash after a move, if the board supports it.
        if hasattr(self.board, 'zobrist_hash'):
            self.zobrist = self.board.zobrist_hash()
        else:
            self.zobrist = hash(self.board.fen())
        # Reset the cached evaluation data since the board has changed.
        self.invalidate_eval()

    def pop(self) -> None:
        """
        Undoes the last move, updates the move history and Zobrist hash, and resets the cached evaluation data.
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
