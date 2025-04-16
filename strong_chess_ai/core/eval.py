"""
Evaluation module for strong_chess_ai.

Provides material, piece-square, pawn structure, mobility, king safety, and rook placement evaluation functions.

All scores are in centipawns (cp), positive for White's advantage, negative for Black's.

>>> import chess
>>> board = chess.Board()
>>> board.set_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
>>> evaluate(board) > 0  # White to move, starting position, should be near 0
True
>>> board.set_fen('8/8/8/8/8/8/5k2/6K1 w - - 0 1')
>>> evaluate(board) > 0  # White up a king
True
"""
import chess
import numpy as np
from typing import Literal

# Base piece values (centipawns)
PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}

# Example PSTs (real engines use more sophisticated tables)
PAWN_PST_MID = np.array([
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0
])
PAWN_PST_END = PAWN_PST_MID
KNIGHT_PST_MID = np.array([
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
])
KNIGHT_PST_END = KNIGHT_PST_MID
BISHOP_PST_MID = np.array([
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
])
BISHOP_PST_END = BISHOP_PST_MID
ROOK_PST_MID = np.array([
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0
])
ROOK_PST_END = ROOK_PST_MID
QUEEN_PST_MID = np.array([
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
])
QUEEN_PST_END = QUEEN_PST_MID
KING_PST_MID = np.array([
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20
])
KING_PST_END = np.array([
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50
])

PST = {
    chess.PAWN:   {"MID": PAWN_PST_MID,   "END": PAWN_PST_END},
    chess.KNIGHT: {"MID": KNIGHT_PST_MID, "END": KNIGHT_PST_END},
    chess.BISHOP: {"MID": BISHOP_PST_MID, "END": BISHOP_PST_END},
    chess.ROOK:   {"MID": ROOK_PST_MID,   "END": ROOK_PST_END},
    chess.QUEEN:  {"MID": QUEEN_PST_MID,  "END": QUEEN_PST_END},
    chess.KING:   {"MID": KING_PST_MID,   "END": KING_PST_END},
}

def material(board: chess.Board) -> int:
    """Return material balance in centipawns (White minus Black)."""
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score

def piece_square(board: chess.Board, phase: Literal["MID", "END"]) -> int:
    """Return piece-square table score (White minus Black) for the given phase."""
    score = 0
    for piece_type in PIECE_VALUES:
        pst = PST[piece_type][phase]
        for square in board.pieces(piece_type, chess.WHITE):
            score += pst[square]
        for square in board.pieces(piece_type, chess.BLACK):
            score -= pst[chess.square_mirror(square)]
    return score

def pawn_bitboards(board: chess.Board, color: chess.Color) -> int:
    """Return a bitmask of the given color's pawns."""
    return board.pawns & board.occupied_co[color]

def _isolated_mask(file: int) -> int:
    """Return bitmask for isolated pawn detection for a file (0=a, 7=h)."""
    mask = 0
    if file > 0:
        mask |= chess.BB_FILES[file - 1]
    if file < 7:
        mask |= chess.BB_FILES[file + 1]
    return mask

def _doubled_count(file: int) -> int:
    """Return the number of extra pawns on a file (doubled pawns count as n-1)."""
    return max(0, bin(chess.BB_FILES[file]).count('1') - 1)

def is_passed(board: chess.Board, sq: int, color: chess.Color) -> bool:
    """Return True if the pawn on sq is a passed pawn for color."""
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    opp_color = not color
    # Passed pawn: no enemy pawns on same or adjacent files ahead
    mask = chess.BB_FILES[file]
    if file > 0:
        mask |= chess.BB_FILES[file - 1]
    if file < 7:
        mask |= chess.BB_FILES[file + 1]
    if color == chess.WHITE:
        in_front = mask & chess.BB_RANKS_MASKS[rank + 1]
        for r in range(rank + 1, 8):
            in_front |= mask & chess.BB_RANKS_MASKS[r]
        return (board.pawns & board.occupied_co[opp_color] & in_front) == 0
    else:
        in_front = mask & chess.BB_RANKS_MASKS[rank - 1]
        for r in range(rank - 1, -1, -1):
            in_front |= mask & chess.BB_RANKS_MASKS[r]
        return (board.pawns & board.occupied_co[opp_color] & in_front) == 0

def pawn_structure(board: chess.Board) -> int:
    """Evaluate pawn structure with refined detection for isolated, doubled, and passed pawns.
    -15 cp per isolated pawn
    -10 cp per doubled pawn (each extra pawn on a file)
    +20/40/80 cp bonus for a passed pawn on rank 5/6/7 (white) or 4/3/2 (black).
    """
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
            # If no friendly pawns on adjacent files
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
    return score

def mobility(board: chess.Board) -> int:
    """Evaluate mobility: number of legal moves (White minus Black)."""
    white_moves = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
    board.push(chess.Move.null())
    black_moves = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
    board.pop()
    return white_moves - black_moves

def king_safety(board: chess.Board, phase: Literal["MID", "END"]) -> int:
    """Evaluate king safety. Penalize exposed kings more in the midgame."""
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is not None:
            # Penalize king on open file
            file = chess.square_file(king_sq)
            open_file = all(board.piece_type_at(chess.square(file, r)) != chess.PAWN for r in range(8))
            penalty = 30 if phase == "MID" else 10
            term = -penalty if open_file else 0
            score += term if color == chess.WHITE else -term
    return score

def rook_placement(board: chess.Board) -> int:
    """Reward rooks on open or semi-open files."""
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
    """Return 'MID' or 'END' depending on non-pawn material left."""
    non_pawn = sum(len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    return "END" if non_pawn <= 6 else "MID"

def evaluate(board: chess.Board) -> int:
    """Evaluate the board and return centipawn score (White minus Black). Applies draw contempt (+/-20 cp)."""
    ph = phase(board)
    score = (
        material(board)
        + piece_square(board, ph)
        + pawn_structure(board)
        + mobility(board)
        + king_safety(board, ph)
        + rook_placement(board)
    )
    # Draw contempt: prefer not to draw
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        score += 20 if board.turn == chess.WHITE else -20
    return int(score)
