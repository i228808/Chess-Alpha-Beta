# Name: Abdullah Mansoor, Roll Number: i228808
"""
Bitboard utilities for Strong Chess AI.

This module provides helper functions to efficiently extract and compute various chess features using bitboards.
The functions included are:
  - count_bits: Counts the number of set bits in a bitboard.
  - squares_in_mask: Returns a list of squares (0-63) in a bitboard.
  - file_mask: Return the bitmask for an entire file (a-h).
  - rank_mask: Return the bitmask for an entire rank (1-8).
  - bitboard_to_array: Convert a bitboard into an 8x8 numpy boolean array.
"""

import chess
import numpy as np

# Central squares: d4, e4, d5, e5 (and extended center)
CENTER_MASK = sum(1 << sq for sq in [chess.D4, chess.E4, chess.D5, chess.E5])
EXT_CENTER_MASK = sum(
    1 << sq
    for sq in [
        chess.C3, chess.C4, chess.C5, chess.C6,
        chess.D3, chess.D4, chess.D5, chess.D6,
        chess.E3, chess.E4, chess.E5, chess.E6,
        chess.F3, chess.F4, chess.F5, chess.F6,
    ]
)

def count_bits(bb: int) -> int:
    """
    Count the number of set bits (1's) in the bitboard (bb).
    
    Args:
        bb (int): Bitboard represented as an integer.
    
    Returns:
        int: Number of bits set to 1.
    """
    return bin(bb).count("1")

def squares_in_mask(bb: int) -> list:
    """
    Return a list of squares (indices 0-63) that are set in the bitboard.
    
    Args:
        bb (int): Bitboard represented as an integer.
    
    Returns:
        list: List of square indices with bits set.
    """
    return [sq for sq in range(64) if (bb >> sq) & 1]

def king_zone_mask(square: int) -> int:
    """
    Compute and return a bitmask representing the king's "zone" – a 3x3 area
    centered on the given square. This is useful for king safety evaluation.
    
    Args:
        square (int): Square index (0-63) where the king is located.
    
    Returns:
        int: Bitboard mask for the king zone.
    
    Example:
        >>> kz = king_zone_mask(chess.E1)
        >>> count_bits(kz)  # For a king in the center, expect 9
        9
    """
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    mask = 0
    for r in range(max(0, rank - 1), min(8, rank + 2)):
        for f in range(max(0, file - 1), min(8, file + 2)):
            mask |= 1 << chess.square(f, r)
    return mask

def file_mask(file_index: int) -> int:
    """
    Return a bitmask for all squares in the given file. Files are numbered 0 through 7,
    corresponding to a through h.
    
    Args:
        file_index (int): File index (0 for file 'a', 7 for file 'h').
    
    Returns:
        int: Bitboard mask for the entire file.
    
    Example:
        >>> fm = file_mask(0)  # File a
        >>> count_bits(fm)
        8
    """
    # chess.BB_FILES is a tuple of bitboards for files a-h
    return chess.BB_FILES[file_index]

def rank_mask(rank_index: int) -> int:
    """
    Return a bitmask for all squares in the given rank. Ranks are numbered 0 through 7,
    where 0 corresponds to rank 1 and 7 corresponds to rank 8.
    
    Args:
        rank_index (int): Rank index (0 for rank 1, 7 for rank 8).
    
    Returns:
        int: Bitboard mask for the entire rank.
    
    Example:
        >>> rm = rank_mask(0)  # Rank 1
        >>> count_bits(rm)
        8
    """
    # chess.BB_RANK_1 corresponds to rank 1; shift it by (rank_index * 8)
    return chess.BB_RANK_1 << (rank_index * 8)

def bitboard_to_array(bb: int) -> np.ndarray:
    """
    Convert a bitboard into an 8x8 NumPy boolean array, where each element is True
    if the corresponding square (by index) is set.
    
    Args:
        bb (int): Bitboard represented as an integer.
    
    Returns:
        np.ndarray: An 8x8 boolean array representation of the bitboard.
    
    Example:
        >>> arr = bitboard_to_array(0xFF)  # Bottom rank full
        >>> arr[0, :]
        array([ True,  True,  True,  True,  True,  True,  True,  True])
    """
    arr = np.zeros((8, 8), dtype=bool)
    for sq in range(64):
        if (bb >> sq) & 1:
            # chess.square_rank(sq) gives 0 at rank 1 and up to 7 at rank 8.
            r = chess.square_rank(sq)
            f = chess.square_file(sq)
            arr[r, f] = True
    return arr

# Additional functions can be added here to support pawn structure evaluation,
# attack generation, and mobility assessment as needed in the evaluation module.
# For example, functions for shifted bitboards to simulate pawn advances or
# combining bitboards for sliding piece rays could be implemented here.
