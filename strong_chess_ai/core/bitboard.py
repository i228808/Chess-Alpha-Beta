"""
Bitboard utilities for fast chess feature extraction (mobility, control, pawn structure, etc).
"""
import chess
import numpy as np

# Central squares: d4, e4, d5, e5 (and extended center)
CENTER_MASK = sum(1 << sq for sq in [chess.D4, chess.E4, chess.D5, chess.E5])
EXT_CENTER_MASK = sum(1 << sq for sq in [chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.D4, chess.D5, chess.D6,
                                         chess.E3, chess.E4, chess.E5, chess.E6, chess.F3, chess.F4, chess.F5, chess.F6])

def count_bits(bb):
    return bin(bb).count('1')

def squares_in_mask(bb):
    return [sq for sq in range(64) if (bb >> sq) & 1]

# Add more as needed for pawn structure, attacks, etc.
