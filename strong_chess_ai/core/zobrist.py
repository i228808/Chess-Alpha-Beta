"""
Zobrist hashing for fast position keys (for TT).
"""
import numpy as np
import chess

np.random.seed(2025)
ZOBRIST_PIECE_KEYS = np.random.randint(0, 2**64, size=(2, 6, 64), dtype=np.uint64)
ZOBRIST_CASTLING_KEYS = np.random.randint(0, 2**64, size=16, dtype=np.uint64)
ZOBRIST_EP_KEYS = np.random.randint(0, 2**64, size=8, dtype=np.uint64)
ZOBRIST_TURN_KEY = np.random.randint(0, 2**64, dtype=np.uint64)

def compute_zobrist(board: chess.Board) -> int:
    key = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            color = int(piece.color)
            ptype = piece.piece_type - 1
            key ^= int(ZOBRIST_PIECE_KEYS[color, ptype, sq])
    key ^= int(ZOBRIST_CASTLING_KEYS[board.castling_rights])
    if board.ep_square is not None:
        key ^= int(ZOBRIST_EP_KEYS[chess.square_file(board.ep_square)])
    if board.turn == chess.BLACK:
        key ^= int(ZOBRIST_TURN_KEY)
    return key
