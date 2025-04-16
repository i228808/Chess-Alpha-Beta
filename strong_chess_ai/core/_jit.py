from numba import njit, int64
import numpy as np

@njit(int64(int64[:], int64[:], int64))
def pst_sum(pieces, pst, color_sign):
    s = 0
    for i in range(pieces.shape[0]):
        s += color_sign * pst[pieces[i]]
    return s

@njit(int64(int64[:], int64[:], int64[:]))
def mobility(white_moves, black_moves, weights):
    # white_moves, black_moves: arrays of move counts per piece
    # weights: optional weighting per piece type
    wm = 0
    bm = 0
    for i in range(white_moves.shape[0]):
        wm += white_moves[i] * weights[i]
    for i in range(black_moves.shape[0]):
        bm += black_moves[i] * weights[i]
    return wm - bm
