"""
Zobrist hashing for fast position keys (for TT).

This module pre-generates 64-bit random keys for each piece (for both colors on each square),
for castling rights, en passant files, and the turn indicator. The compute_zobrist() function 
quickly computes a unique hash for a given board position using these keys.

The generated keys remain constant during a program run (seeded deterministically for reproducibility).
"""

import numpy as np
import chess
from typing import Any

# Set a deterministic random seed for reproducibility.
np.random.seed(2025)

# Pre-generate Zobrist keys:
# Dimensions: color (2), piece type (6), squares (64)
ZOBRIST_PIECE_KEYS = np.random.randint(0, 2**64, size=(2, 6, 64), dtype=np.uint64)
# Castling rights: there are 16 possible castling rights combinations.
ZOBRIST_CASTLING_KEYS = np.random.randint(0, 2**64, size=16, dtype=np.uint64)
# En passant: one key per file (files a-h).
ZOBRIST_EP_KEYS = np.random.randint(0, 2**64, size=8, dtype=np.uint64)
# Turn key: applied if it is Black's turn.
ZOBRIST_TURN_KEY = np.random.randint(0, 2**64, dtype=np.uint64)

def compute_zobrist(board: chess.Board) -> int:
    """
    Compute and return the Zobrist hash key for the given board position.

    The Zobrist key is a 64-bit integer computed by XOR-ing:
      - A random key for each piece on a given square,
      - A key for the current castling rights,
      - A key for the en passant square (if any),
      - And a key for Black's turn (applied only when board.turn is Black).

    This function uses board.piece_map() to iterate only over squares that are occupied,
    resulting in faster computation on sparse boards typical in chess.
    
    Args:
        board (chess.Board): The current chess board state.
    
    Returns:
        int: The 64-bit Zobrist hash for the board position.
    
    Example:
        >>> import chess
        >>> board = chess.Board()
        >>> key = compute_zobrist(board)
        >>> isinstance(key, int)
        True
    """
    key = 0
    # Iterate over all pieces on the board for efficiency.
    for sq, piece in board.piece_map().items():
        color_index: int = int(piece.color)  # 0 for White, 1 for Black.
        ptype_index: int = piece.piece_type - 1  # Piece types 1..6 become 0..5.
        key ^= int(ZOBRIST_PIECE_KEYS[color_index, ptype_index, sq])
    
    # Incorporate castling rights. board.castling_rights is an int in [0, 15].
    key ^= int(ZOBRIST_CASTLING_KEYS[board.castling_rights])
    
    # Incorporate en passant key if there's an en passant square.
    if board.ep_square is not None:
        file_idx: int = chess.square_file(board.ep_square)
        key ^= int(ZOBRIST_EP_KEYS[file_idx])
    
    # Incorporate turn: if it is Black's turn, XOR with the turn key.
    if board.turn == chess.BLACK:
        key ^= int(ZOBRIST_TURN_KEY)
    
    return key
