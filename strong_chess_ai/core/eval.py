"""
Evaluation module for strong_chess_ai.

This module implements a multi-component, phase‑tapered evaluation function.
It computes a 20-dimensional feature vector based on:
  f0  - Material balance (in centipawns)
  f1  - Mobility (difference in legal moves × 10)
  f2  - Central occupation (pieces on d4, e4, d5, e5)
  f3  - Central control (attacks on central squares)
  f4  - Space advantage (control on opponent’s half)
  f5  - Pawn shield (support around king)
  f6  - Open file penalty for king safety
  f7  - Tropism (enemy piece proximity to king)
  f8  - King zone attacks (attacks into a 3x3 king zone)
  f9  - Isolated pawns count (penalty)
  f10 - Doubled pawns count (penalty)
  f11 - Passed pawns count (bonus)
  f12 - Backward pawns count (penalty)
  f13 - Pawn islands count (penalty per extra island)
  f14 - Knight outpost bonus
  f15 - Bishop pair bonus
  f16 - Hanging pieces count (penalty)
  f17 - Absolute pins count (penalty)
  f18 - Mate threat count (bonus if you threaten mate)
  f19 - Fork potential (bonus if multiple valuable targets are attacked)

The final evaluation score is computed by blending midgame (W_MID) and endgame (W_END)
weights according to a phase factor φ ∈ [0,1] (φ=1 for midgame, φ=0 for endgame).

All scores are expressed in centipawns (cp), with positive values indicating an advantage for White.
    
>>> import chess
>>> board = chess.Board()
>>> eval_score = tapered_eval(board)
>>> isinstance(eval_score, float)
True
"""

import chess
import numpy as np
from typing import Literal

def material(board: chess.Board) -> int:
    """
    Return material balance in centipawns (White minus Black).
    """
    return sum(PIECE_VALUES_CP[pt] * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
               for pt in PIECE_VALUES_CP)


# -------------------------------
# Base piece values and PST arrays
# -------------------------------
PIECE_VALUES_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900
}
# For King we only use PST; its material value is excluded.
# Sample Piece-Square Tables (PSTs) – these are examples; real tuning is advised.
PAWN_PST_MID = np.array([
     0,   0,   0,   0,   0,   0,   0,   0,
     5,  10,  10, -20, -20,  10,  10,   5,
     5,  -5, -10,   0,   0, -10,  -5,   5,
     0,   0,   0,  20,  20,   0,   0,   0,
     5,   5,  10,  25,  25,  10,   5,   5,
    10,  10,  20,  30,  30,  20,  10,  10,
    50,  50,  50,  50,  50,  50,  50,  50,
     0,   0,   0,   0,   0,   0,   0,   0
])
PAWN_PST_END = PAWN_PST_MID.copy()

KNIGHT_PST_MID = np.array([
   -50, -40, -30, -30, -30, -30, -40, -50,
   -40, -20,   0,   0,   0,   0, -20, -40,
   -30,   0,  10,  15,  15,  10,   0, -30,
   -30,   5,  15,  20,  20,  15,   5, -30,
   -30,   0,  15,  20,  20,  15,   0, -30,
   -30,   5,  10,  15,  15,  10,   5, -30,
   -40, -20,   0,   5,   5,   0, -20, -40,
   -50, -40, -30, -30, -30, -30, -40, -50
])
KNIGHT_PST_END = KNIGHT_PST_MID.copy()

BISHOP_PST_MID = np.array([
   -20, -10, -10, -10, -10, -10, -10, -20,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -10,   0,   5,  10,  10,   5,   0, -10,
   -10,   5,   5,  10,  10,   5,   5, -10,
   -10,   0,  10,  10,  10,  10,   0, -10,
   -10,  10,  10,  10,  10,  10,  10, -10,
   -10,   5,   0,   0,   0,   0,   5, -10,
   -20, -10, -10, -10, -10, -10, -10, -20
])
BISHOP_PST_END = BISHOP_PST_MID.copy()

ROOK_PST_MID = np.array([
     0,   0,   0,   5,   5,   0,   0,   0,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
     5,  10,  10,  10,  10,  10,  10,   5,
     0,   0,   0,   0,   0,   0,   0,   0
])
ROOK_PST_END = ROOK_PST_MID.copy()

QUEEN_PST_MID = np.array([
   -20, -10, -10,  -5,  -5, -10, -10, -20,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -10,   0,   5,   5,   5,   5,   0, -10,
    -5,   0,   5,   5,   5,   5,   0,  -5,
     0,   0,   5,   5,   5,   5,   0,  -5,
   -10,   5,   5,   5,   5,   5,   0, -10,
   -10,   0,   5,   0,   0,   0,   0, -10,
   -20, -10, -10,  -5,  -5, -10, -10, -20
])
QUEEN_PST_END = QUEEN_PST_MID.copy()

KING_PST_MID = np.array([
   -30, -40, -40, -50, -50, -40, -40, -30,
   -30, -40, -40, -50, -50, -40, -40, -30,
   -30, -40, -40, -50, -50, -40, -40, -30,
   -30, -40, -40, -50, -50, -40, -40, -30,
   -20, -30, -30, -40, -40, -30, -30, -20,
   -10, -20, -20, -20, -20, -20, -20, -10,
    20,  20,   0,   0,   0,   0,  20,  20,
    20,  30,  10,   0,   0,  10,  30,  20
])
KING_PST_END = np.array([
   -50, -40, -30, -20, -20, -30, -40, -50,
   -30, -20, -10,   0,   0, -10, -20, -30,
   -30, -10,  20,  30,  30,  20, -10, -30,
   -30, -10,  30,  40,  40,  30, -10, -30,
   -30, -10,  30,  40,  40,  30, -10, -30,
   -30, -10,  20,  30,  30,  20, -10, -30,
   -30, -30,   0,   0,   0,   0, -30, -30,
   -50, -30, -30, -30, -30, -30, -30, -50
])
# A simple king centralisation table (endgame), using Manhattan distance from center squares (d4, e4, d5, e5)
KING_ENDGAME_TABLE = np.array([
    3, 2, 2, 2, 2, 2, 2, 3,
    2, 1, 1, 1, 1, 1, 1, 2,
    2, 1, 0, 0, 0, 0, 1, 2,
    2, 1, 0, 0, 0, 0, 1, 2,
    2, 1, 0, 0, 0, 0, 1, 2,
    2, 1, 0, 0, 0, 0, 1, 2,
    2, 1, 1, 1, 1, 1, 1, 2,
    3, 2, 2, 2, 2, 2, 2, 3
])

# Pre-built PST dictionaries for use in specialized functions
PST_MID = {
    chess.PAWN:   PAWN_PST_MID,
    chess.KNIGHT: KNIGHT_PST_MID,
    chess.BISHOP: BISHOP_PST_MID,
    chess.ROOK:   ROOK_PST_MID,
    chess.QUEEN:  QUEEN_PST_MID,
    chess.KING:   KING_PST_MID,
}
PST_END = {
    chess.PAWN:   PAWN_PST_END,
    chess.KNIGHT: KNIGHT_PST_END,
    chess.BISHOP: BISHOP_PST_END,
    chess.ROOK:   ROOK_PST_END,
    chess.QUEEN:  QUEEN_PST_END,
    chess.KING:   KING_PST_END,
}

# -------------------------------
# Tapered Feature-Vector Evaluation (V3)
# -------------------------------
# Example weight vectors (20 dimensions); these should be tuned.
W_MID = np.array([
     1.0, 10.0, 12.0,  8.0,  7.0,
    15.0, -10.0, -5.0,  8.0, -15.0,
   -10.0, 25.0, -10.0,  -8.0, 20.0,
    30.0, -25.0, -15.0, 100.0, 18.0
])
W_END = np.array([
     1.0,  8.0,  8.0,  6.0,  5.0,
    10.0,  -8.0,  -3.0,  6.0, -10.0,
    -7.0, 30.0,  -8.0,  -5.0, 15.0,
    35.0, -15.0, -10.0, 100.0, 10.0
])

# Precomputed masks and constants
CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
CENTER_MASK = sum(1 << sq for sq in CENTER_SQUARES)
# King zone: 3x3 area around the king.
KING_ZONE_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

# -------------------------------
# Phase Computation
# -------------------------------
def compute_phase(board: chess.Board) -> float:
    """
    Compute the phase factor φ ∈ [0, 1] where 1.0 indicates full midgame and 0.0 indicates endgame.
    We compute this by evaluating non-pawn material on the board.
    """
    # Maximum phase corresponds roughly to a full complement of non-pawn pieces.
    # Here we use a simple normalized sum of queen, rook, bishop, and knight counts.
    max_phase = 2 * 900 + 4 * 500 + 4 * 330 + 4 * 320  # example upper bound in centipawns
    material_left = (
        PIECE_VALUES_CP[chess.QUEEN] * (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))) +
        PIECE_VALUES_CP[chess.ROOK]   * (len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))) +
        PIECE_VALUES_CP[chess.BISHOP] * (len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))) +
        PIECE_VALUES_CP[chess.KNIGHT] * (len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)))
    )
    phase = material_left / max_phase
    return min(1.0, max(0.0, phase))

# -------------------------------
# Feature Extraction
# -------------------------------
def feature_vector(board: chess.Board) -> np.ndarray:
    """
    Extract a 20-dimensional feature vector for the given board.
    Feature indices:
      0: Material balance (in cp)
      1: Mobility (difference in legal moves * 10)
      2: Center occupation (count of pieces on d4,e4,d5,e5: white +1, black -1)
      3: Central control (difference in attacks on center squares)
      4: Space advantage (control on opponent half)
      5: Pawn shield (for kings, supported pawns)
      6: Open file penalty (if king on file without own pawn)
      7: Tropism (sum_{enemy pieces} [value/distance] relative to opponent king)
      8: King zone attacks (attacks into 3x3 area around own king)
      9: Isolated pawns (count, penalty)
     10: Doubled pawns (extra pawns per file)
     11: Passed pawns (count)
     12: Backward pawns (count)
     13: Pawn islands (count)
     14: Knight outpost (count)
     15: Bishop pair (bonus: 1 if present)
     16: Hanging pieces (count of pieces attacked by opponent and not defended)
     17: Absolute pins (count)
     18: Mate threats (number of moves that deliver checkmate)
     19: Fork potential (count of knight forks yielding multiple targets)
    Returns:
        np.ndarray: Feature vector (shape: (20,))
    """
    f = np.zeros(20)
    # f0: Material balance (in centipawns)
    f[0] = sum(PIECE_VALUES_CP[pt] * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
               for pt in PIECE_VALUES_CP)
    
    # f1: Mobility. (Note: using board.legal_moves.count() for current turn)
    white_moves = sum(1 for _ in board.legal_moves) if board.turn == chess.WHITE else 0
    # To approximate Black's mobility, push a null move temporarily.
    board.push(chess.Move.null())
    black_moves = sum(1 for _ in board.legal_moves) if board.turn == chess.BLACK else 0
    board.pop()
    f[1] = (white_moves - black_moves) * 10

    # f2: Center occupation: +1 per White piece on center squares, -1 per Black piece.
    occ = 0
    for sq in CENTER_SQUARES:
        piece = board.piece_at(sq)
        if piece:
            occ += 1 if piece.color == chess.WHITE else -1
    f[2] = occ

    # f3: Central control: count attacks on the CENTER_MASK.
    white_attacks = 0
    black_attacks = 0
    for color in [chess.WHITE, chess.BLACK]:
        for pt in PIECE_VALUES_CP:
            for sq in board.pieces(pt, color):
                # board.attacks(sq) returns a set of attacked squares.
                if color == chess.WHITE:
                    white_attacks += sum(1 for a in board.attacks(sq) if ((1 << a) & CENTER_MASK))
                else:
                    black_attacks += sum(1 for a in board.attacks(sq) if ((1 << a) & CENTER_MASK))
    f[3] = white_attacks - black_attacks

    # f4: Space advantage: count squares (e.g. on opponent half) attacked.
    white_space = 0
    black_space = 0
    for sq in range(64):
        rank = chess.square_rank(sq)
        if rank >= 4 and (any(sq in board.attacks(sq2) for sq2 in board.pieces(chess.PAWN, chess.WHITE))):
            white_space += 1
        if rank <= 3 and (any(sq in board.attacks(sq2) for sq2 in board.pieces(chess.PAWN, chess.BLACK))):
            black_space += 1
    f[4] = white_space - black_space

    # f5: Pawn shield: for each king, count friendly pawn(s) in the two ranks ahead of king.
    shield_white = 0
    shield_black = 0
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is not None:
            file = chess.square_file(king_sq)
            rank = chess.square_rank(king_sq)
            shield = 0
            for df in [-1, 0, 1]:
                f_file = file + df
                if 0 <= f_file < 8:
                    for dr in [1, 2]:
                        r = rank + (dr if color == chess.WHITE else -dr)
                        if 0 <= r < 8:
                            sq_idx = chess.square(f_file, r)
                            piece = board.piece_at(sq_idx)
                            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                                shield += 1
            if color == chess.WHITE:
                shield_white = shield
            else:
                shield_black = shield
    f[5] = shield_white - shield_black

    # f6: Open file penalty: if king is on a file with no friendly pawn.
    open_file_white = 0
    open_file_black = 0
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is not None:
            file = chess.square_file(king_sq)
            pawns = board.pieces(chess.PAWN, color)
            files = [chess.square_file(sq) for sq in pawns]
            penalty = 1 if files.count(file) == 0 else 0
            if color == chess.WHITE:
                open_file_white = penalty
            else:
                open_file_black = penalty
    f[6] = -open_file_white + open_file_black

    # f7: Tropism. For each color, sum (piece value/distance) from enemy king.
    tropism_white = 0
    tropism_black = 0
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    for color in [chess.WHITE, chess.BLACK]:
        opp_king = black_king if color == chess.WHITE else white_king
        if opp_king is None:
            continue
        for pt in PIECE_VALUES_CP:
            for sq in board.pieces(pt, color):
                dist = abs(chess.square_file(sq) - chess.square_file(opp_king)) + abs(chess.square_rank(sq) - chess.square_rank(opp_king))
                if dist > 0:
                    if color == chess.WHITE:
                        tropism_white += PIECE_VALUES_CP[pt] / dist
                    else:
                        tropism_black += PIECE_VALUES_CP[pt] / dist
    f[7] = tropism_white - tropism_black

    # f8: King zone attacks: count opponent attacks in a 3x3 block around each king.
    zone_attack_white = 0
    zone_attack_black = 0
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            continue
        zone = set()
        kf = chess.square_file(king_sq)
        kr = chess.square_rank(king_sq)
        for df, dr in KING_ZONE_OFFSETS:
            nf = kf + df
            nr = kr + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                zone.add(chess.square(nf, nr))
        opp = not color
        count = sum(1 for sq in zone if board.is_attacked_by(opp, sq))
        if color == chess.WHITE:
            zone_attack_white = count
        else:
            zone_attack_black = count
    f[8] = -zone_attack_white + zone_attack_black

    # f9: Isolated pawns: count number of pawns with no friendly pawn on adjacent files.
    iso_white = 0
    iso_black = 0
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        files = [chess.square_file(sq) for sq in pawns]
        for sq in pawns:
            f_file = chess.square_file(sq)
            left = f_file - 1
            right = f_file + 1
            if (left < 0 or left not in files) and (right > 7 or right not in files):
                if color == chess.WHITE:
                    iso_white += 1
                else:
                    iso_black += 1
    f[9] = iso_white - iso_black

    # f10: Doubled pawns: count extra pawns in a file.
    doub_white = 0
    doub_black = 0
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        file_counts = {}
        for sq in pawns:
            f_file = chess.square_file(sq)
            file_counts[f_file] = file_counts.get(f_file, 0) + 1
        extra = sum(max(0, cnt - 1) for cnt in file_counts.values())
        if color == chess.WHITE:
            doub_white = extra
        else:
            doub_black = extra
    f[10] = doub_white - doub_black

    # f11: Passed pawns: count pawns with no opposing pawn on same/adjacent files ahead.
    passed_white = 0
    passed_black = 0
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        opp_pawns = board.pieces(chess.PAWN, not color)
        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            is_passed = True
            for df in [-1, 0, 1]:
                f_file = file + df
                if 0 <= f_file < 8:
                    for opp_sq in opp_pawns:
                        if chess.square_file(opp_sq) == f_file:
                            opp_rank = chess.square_rank(opp_sq)
                            if (color == chess.WHITE and opp_rank > rank) or (color == chess.BLACK and opp_rank < rank):
                                is_passed = False
            if is_passed:
                if color == chess.WHITE:
                    passed_white += 1
                else:
                    passed_black += 1
    f[11] = passed_white - passed_black

    # f12: Backward pawns: count pawn that cannot be supported by a friendly pawn from behind.
    backward_white = 0
    backward_black = 0
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        opp_pawns = board.pieces(chess.PAWN, not color)
        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            if color == chess.WHITE:
                if all((chess.square_file(opp_sq) != file or chess.square_rank(opp_sq) <= rank) for opp_sq in opp_pawns):
                    backward_white += 1
            else:
                if all((chess.square_file(opp_sq) != file or chess.square_rank(opp_sq) >= rank) for opp_sq in opp_pawns):
                    backward_black += 1
    f[12] = backward_white - backward_black

    # f13: Pawn islands: count groups of adjacent files that contain at least one pawn.
    islands_white = 0
    islands_black = 0
    for color in [chess.WHITE, chess.BLACK]:
        pawns = sorted(board.pieces(chess.PAWN, color))
        prev_file = -2
        islands = 0
        for sq in pawns:
            file = chess.square_file(sq)
            if file != prev_file + 1:
                islands += 1
            prev_file = file
        if color == chess.WHITE:
            islands_white = islands
        else:
            islands_black = islands
    f[13] = islands_white - islands_black

    # f14: Knight outpost: count knight or bishop outposts (supported and not attacked by enemy pawn).
    outpost_white = 0
    outpost_black = 0
    def is_valid_square(sq):
        return 0 <= sq < 64
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for sq in board.pieces(piece_type, color):
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                if color == chess.WHITE:
                    support = ((is_valid_square(sq - 9) and (sq - 9) in pawns) or (is_valid_square(sq - 7) and (sq - 7) in pawns))
                    safe = not ((is_valid_square(sq + 7) and (sq + 7) in board.pieces(chess.PAWN, not color)) or (is_valid_square(sq + 9) and (sq + 9) in board.pieces(chess.PAWN, not color)))
                    if support and safe:
                        outpost_white += 1
                else:
                    support = ((is_valid_square(sq + 9) and (sq + 9) in pawns) or (is_valid_square(sq + 7) and (sq + 7) in pawns))
                    safe = not ((is_valid_square(sq - 7) and (sq - 7) in board.pieces(chess.PAWN, not color)) or (is_valid_square(sq - 9) and (sq - 9) in board.pieces(chess.PAWN, not color)))
                    if support and safe:
                        outpost_black += 1
    f[14] = outpost_white - outpost_black

    # f15: Bishop pair: bonus if a side has two or more bishops.
    bp_white = 1 if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2 else 0
    bp_black = 1 if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2 else 0
    f[15] = bp_white - bp_black

    # f16: Hanging pieces: count pieces that are attacked by the opponent and not defended.
    hanging_white = 0
    hanging_black = 0
    for color in [chess.WHITE, chess.BLACK]:
        opp = not color
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                if board.is_attacked_by(opp, sq) and not board.is_attacked_by(color, sq):
                    if color == chess.WHITE:
                        hanging_white += 1
                    else:
                        hanging_black += 1
    f[16] = hanging_white - hanging_black

    # f17: Absolute pins: count pieces that are pinned (excluding the king)
    pin_white = 0
    pin_black = 0
    for color in [chess.WHITE, chess.BLACK]:
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                if board.is_pinned(color, sq):
                    if color == chess.WHITE:
                        pin_white += 1
                    else:
                        pin_black += 1
    f[17] = pin_white - pin_black

    # f18: Mate threat: count moves that immediately deliver checkmate.
    mate_white = 0
    mate_black = 0
    for color in [chess.WHITE, chess.BLACK]:
        count = 0
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                count += 1
            board.pop()
        if color == chess.WHITE:
            mate_white = count
        else:
            mate_black = count
    f[18] = mate_white - mate_black

    # f19: Fork potential: count knights with potential to fork two or more high‑value targets.
    fork_white = 0
    fork_black = 0
    for color in [chess.WHITE, chess.BLACK]:
        for sq in board.pieces(chess.KNIGHT, color):
            targets = list(board.attacks(sq))
            valuable = [t for t in targets if board.piece_at(t) and board.piece_at(t).color != color and board.piece_at(t).piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP)]
            if len(valuable) >= 2:
                if color == chess.WHITE:
                    fork_white += 1
                else:
                    fork_black += 1
    f[19] = fork_white - fork_black

    return f

def tapered_eval(board: chess.Board) -> float:
    """
    Returns the tapered evaluation score (centipawns) computed by blending midgame and endgame evaluations.
    
    It computes the feature vector f, gets the phase factor φ, and returns:
      score = φ * dot(W_MID, f) + (1 - φ) * dot(W_END, f)
    
    >>> import chess
    >>> board = chess.Board()
    >>> isinstance(tapered_eval(board), float)
    True
    """
    f = feature_vector(board)
    phi = compute_phase(board)
    return float(phi * np.dot(W_MID, f) + (1 - phi) * np.dot(W_END, f))

# -------------------------------
# Additional (Legacy) Evaluation Functions
# These functions remain available for comparison or fallback.
# -------------------------------

def pawn_structure(board: chess.Board) -> int:
    """
    Evaluate pawn structure with refined detection for isolated, doubled, and passed pawns.
    
    -15 cp per isolated pawn, -10 cp per extra pawn on a file, 
    and a bonus (+20, +40, +80 cp) for a passed pawn on advanced ranks.
    Uses a persistent cache keyed by (white_pawns, black_pawns) bitboards.
    """
    wp = int(board.pawns & board.occupied_co[chess.WHITE])
    bp = int(board.pawns & board.occupied_co[chess.BLACK])
    key = (wp, bp)
    if key in pawn_cache:
        return pawn_cache[key]
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        pawn_files = [chess.square_file(sq) for sq in pawns]
        file_counts = {f: pawn_files.count(f) for f in set(pawn_files)}
        # Isolated pawns
        isolated = 0
        for sq in pawns:
            file = chess.square_file(sq)
            mask = _isolated_mask(file)
            if not any(sq2 for sq2 in pawns if chess.square_file(sq2) != file and (chess.BB_SQUARES[sq2] & mask)):
                isolated += 1
        # Doubled pawns
        doubled = sum(max(0, count - 1) for count in file_counts.values())
        # Passed pawns
        passed_bonus = 0
        for sq in pawns:
            if is_passed(board, sq, color):
                rank = chess.square_rank(sq) if color == chess.WHITE else 7 - chess.square_rank(sq)
                if rank == 4:
                    passed_bonus += 20
                elif rank == 5:
                    passed_bonus += 40
                elif rank == 6:
                    passed_bonus += 80
        term = -15 * isolated - 10 * doubled + passed_bonus
        score += term if color == chess.WHITE else -term
    pawn_cache[key] = score
    return score

def mobility(board: chess.Board) -> int:
    """Evaluate mobility: difference in the number of legal moves (White minus Black)."""
    white_moves = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
    board.push(chess.Move.null())
    black_moves = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
    board.pop()
    return white_moves - black_moves

def king_safety(board: chess.Board, phase: Literal["MID", "END"]) -> int:
    """Evaluate king safety by penalizing a king on an open file.
    Heavier penalties are applied in the midgame.
    """
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is not None:
            file = chess.square_file(king_sq)
            open_file = all(board.piece_type_at(chess.square(file, r)) != chess.PAWN for r in range(8))
            penalty = 30 if phase == "MID" else 10
            score += -penalty if color == chess.WHITE else penalty
    return score

def rook_placement(board: chess.Board) -> int:
    """Reward rooks positioned on open or semi-open files."""
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        rooks = board.pieces(chess.ROOK, color)
        for sq in rooks:
            file = chess.square_file(sq)
            pawns = [chess.square(file, r) for r in range(8)]
            pawn_count = sum(1 for s in pawns if board.piece_type_at(s) == chess.PAWN and board.color_at(s) == color)
            if pawn_count == 0:
                score += 15 if color == chess.WHITE else -15
            elif pawn_count == 1:
                score += 7 if color == chess.WHITE else -7
    return score

def phase(board: chess.Board) -> Literal["MID", "END"]:
    """Return 'MID' if non-pawn material exceeds a threshold, else 'END'."""
    non_pawn = sum(len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)) 
                   for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    return "END" if non_pawn <= 6 else "MID"

def evaluate_endgame(board: chess.Board) -> int:
    """
    Specialized endgame evaluation that rewards king centralisation,
    active passed pawns, and detects drawn endings.
    """
    # Detect drawn endgames
    pieces = [pt for pt in range(1, 7) for _ in board.pieces(pt, chess.WHITE)] + \
             [pt for pt in range(1, 7) for _ in board.pieces(pt, chess.BLACK)]
    if pieces == [chess.KING, chess.KING]:
        return 0
    if sorted(pieces) in ([chess.KING, chess.KING, chess.KNIGHT],
                          [chess.KING, chess.KING, chess.BISHOP],
                          [chess.KING, chess.KING, chess.KNIGHT, chess.KNIGHT]):
        return 0
    score = 0
    # King activity bonus: reward kings closer to center (using KING_ENDGAME_TABLE)
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is not None:
            bonus = -5 * KING_ENDGAME_TABLE[king_sq]
            score += bonus if color == chess.WHITE else -bonus
    score += material(board)
    score += piece_square(board, "END")
    # Passed pawns scaled
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        for sq in pawns:
            if is_passed(board, sq, color):
                rank = chess.square_rank(sq) if color == chess.WHITE else 7 - chess.square_rank(sq)
                if rank == 4:
                    score += (30 if color == chess.WHITE else -30)
                elif rank == 5:
                    score += (60 if color == chess.WHITE else -60)
                elif rank == 6:
                    score += (120 if color == chess.WHITE else -120)
    return score

def tapered_weight(board: chess.Board) -> float:
    """
    Returns a weight factor for tapering evaluation between midgame and endgame.
    1.0 indicates full midgame; 0.0 indicates full endgame.
    """
    max_phase = 4 * PIECE_VALUES_CP[chess.QUEEN] + 4 * PIECE_VALUES_CP[chess.ROOK] + 4 * PIECE_VALUES_CP[chess.BISHOP] + 4 * PIECE_VALUES_CP[chess.KNIGHT]
    phase_score = (
        PIECE_VALUES_CP[chess.QUEEN] * (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))) +
        PIECE_VALUES_CP[chess.ROOK]   * (len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))) +
        PIECE_VALUES_CP[chess.BISHOP] * (len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))) +
        PIECE_VALUES_CP[chess.KNIGHT] * (len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)))
    )
    return min(1.0, max(0.0, phase_score / max_phase))

def bishop_pair(board: chess.Board) -> int:
    """Return a bonus (±30) if a side has two or more bishops."""
    bonus = 0
    for color in [chess.WHITE, chess.BLACK]:
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            bonus += 30 if color == chess.WHITE else -30
    return bonus

def central_control(board: chess.Board) -> int:
    """
    Evaluate central control: white earns bonus for attacking central squares;
    black loses equivalent points.
    """
    from .bitboard import CENTER_MASK, count_bits
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        attacks = 0
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for sq in board.pieces(pt, color):
                attacks |= int(board.attacks(sq))
        central = count_bits(attacks & CENTER_MASK)
        bonus = 10 * central
        score += bonus if color == chess.WHITE else -bonus
    return score

def space_advantage(board: chess.Board) -> int:
    """
    Evaluate space advantage: count pawns advanced into opponent's territory.
    """
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        for sq in pawns:
            rank = chess.square_rank(sq)
            if (color == chess.WHITE and rank >= 4) or (color == chess.BLACK and rank <= 3):
                score += 8 if color == chess.WHITE else -8
    return score

def outpost_bonus(board: chess.Board) -> int:
    """
    Evaluate knight or bishop outposts: bonus if the piece is on an outpost
    square (supported by a pawn and not attackable by enemy pawns).
    """
    score = 0
    def is_valid_square(sq):
        return 0 <= sq < 64
    for color in [chess.WHITE, chess.BLACK]:
        opp = not color
        pawns = board.pieces(chess.PAWN, color)
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for sq in board.pieces(piece_type, color):
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                if color == chess.WHITE:
                    support = ((is_valid_square(sq - 9) and (sq - 9) in pawns) or 
                               (is_valid_square(sq - 7) and (sq - 7) in pawns))
                    safe = not ((is_valid_square(sq + 7) and (sq + 7) in board.pieces(chess.PAWN, opp)) or 
                                (is_valid_square(sq + 9) and (sq + 9) in board.pieces(chess.PAWN, opp)))
                    if support and safe:
                        score += 18
                else:
                    support = ((is_valid_square(sq + 9) and (sq + 9) in pawns) or 
                               (is_valid_square(sq + 7) and (sq + 7) in pawns))
                    safe = not ((is_valid_square(sq - 7) and (sq - 7) in board.pieces(chess.PAWN, opp)) or 
                                (is_valid_square(sq - 9) and (sq - 9) in board.pieces(chess.PAWN, opp)))
                    if support and safe:
                        score -= 18
    return score

def tactical_features(board: chess.Board) -> int:
    """
    Evaluate tactical features:
      - Hanging pieces: penalize if attacked by opponent and not defended.
      - (Future work: Pins, forks, and mate threats can be added here.)
    """
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        opp = not color
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                if board.is_attacked_by(opp, sq) and not board.is_attacked_by(color, sq):
                    val = PIECE_VALUES_CP.get(piece.piece_type, 0) // 8
                    score += -val if color == chess.WHITE else val
    return score

def evaluate(board: chess.Board) -> int:
    """
    Tapered, multi-component evaluation function for Search Engine V3.
    Combines midgame and endgame evaluations by weighting features based on game phase.
    
    The function blends the midgame (mg) and endgame (eg) scores:
      score = φ * mg + (1 - φ) * eg,
    where φ is computed from non-pawn material remaining.
    
    It also applies a small draw contempt bonus.
    """
    w = tapered_weight(board)
    mg = (
        material(board) +
        bishop_pair(board) +
        piece_square(board, "MID") +
        pawn_structure(board) +
        mobility(board) +
        king_safety(board, "MID") +
        rook_placement(board) +
        central_control(board) +
        space_advantage(board) +
        outpost_bonus(board) +
        tactical_features(board)
    )
    eg = (
        material(board) +
        bishop_pair(board) +
        piece_square(board, "END") +
        pawn_structure(board) +
        mobility(board) +
        king_safety(board, "END") +
        rook_placement(board) +
        central_control(board) +
        space_advantage(board) +
        outpost_bonus(board) +
        tactical_features(board)
    )
    score = int(w * mg + (1 - w) * eg)
    # Draw contempt: adjust score to discourage draws
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        score += 20 if board.turn == chess.WHITE else -20
    return score
