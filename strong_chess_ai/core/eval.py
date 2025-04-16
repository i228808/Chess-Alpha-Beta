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
# King centralisation for endgame (distance from center: e4,d4,e5,d5)
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
        in_front = mask & chess.BB_RANK_MASKS[rank + 1]
        for r in range(rank + 1, 8):
            in_front |= mask & chess.BB_RANK_MASKS[r]
        return (board.pawns & board.occupied_co[opp_color] & in_front) == 0
    else:
        in_front = mask & chess.BB_RANK_MASKS[rank - 1]
        for r in range(rank - 1, -1, -1):
            in_front |= mask & chess.BB_RANK_MASKS[r]
        return (board.pawns & board.occupied_co[opp_color] & in_front) == 0

# Global pawn structure cache: {(white_pawns, black_pawns): score}
pawn_cache = {}

def pawn_structure(board: chess.Board) -> int:
    """Evaluate pawn structure with refined detection for isolated, doubled, and passed pawns.
    Uses a persistent cache keyed by (white_pawns, black_pawns) bitboards.
    -15 cp per isolated pawn
    -10 cp per doubled pawn (each extra pawn on a file)
    +20/40/80 cp bonus for a passed pawn on rank 5/6/7 (white) or 4/3/2 (black).
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
    pawn_cache[key] = score
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

def evaluate_endgame(board: chess.Board) -> int:
    """Evaluate endgame: king activity, scaled passed pawns, drawn endings (K vs K, KB vs K, KN vs K, KNN vs K)."""
    # Detect basic drawn endgames
    pieces = [pt for pt in range(1, 7) for _ in board.pieces(pt, chess.WHITE)] + [pt for pt in range(1, 7) for _ in board.pieces(pt, chess.BLACK)]
    # Only kings
    if pieces == [chess.KING, chess.KING]:
        return 0
    # K vs K+N, K vs K+B, K vs K+NN
    if sorted(pieces) in ([chess.KING, chess.KING, chess.KNIGHT], [chess.KING, chess.KING, chess.BISHOP], [chess.KING, chess.KING, chess.KNIGHT, chess.KNIGHT]):
        return 0
    score = 0
    # King activity bonus: -5cp × distance from centre
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is not None:
            dist = KING_ENDGAME_TABLE[king_sq]
            bonus = -5 * dist
            score += bonus if color == chess.WHITE else -bonus
    # Material and PST
    score += material(board)
    score += piece_square(board, "END")
    # Passed pawns (scaled)
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

    pieces = [pt for pt in range(1, 7) for _ in board.pieces(pt, chess.WHITE)] + [pt for pt in range(1, 7) for _ in board.pieces(pt, chess.BLACK)]
    # Only kings
    if pieces == [chess.KING, chess.KING]:
        return 0
    # K vs K+N, K vs K+B, K vs K+NN
    if sorted(pieces) in ([chess.KING, chess.KING, chess.KNIGHT], [chess.KING, chess.KING, chess.BISHOP], [chess.KING, chess.KING, chess.KNIGHT, chess.KNIGHT]):
        return 0
    score = 0
    # King activity bonus: -5cp × distance from centre
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is not None:
            dist = KING_ENDGAME_TABLE[king_sq]
            bonus = -5 * dist
            score += bonus if color == chess.WHITE else -bonus
    # Material and PST
    score += material(board)
    score += piece_square(board, "END")
    # Passed pawns (scaled)
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
    """Return game phase weight for tapering: 1.0 = midgame, 0.0 = endgame."""
    # Use non-pawn material left (scaled)
    max_phase = 4 * PIECE_VALUES[chess.QUEEN] + 4 * PIECE_VALUES[chess.ROOK] + 4 * PIECE_VALUES[chess.BISHOP] + 4 * PIECE_VALUES[chess.KNIGHT]
    phase_score = (
        PIECE_VALUES[chess.QUEEN] * (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))) +
        PIECE_VALUES[chess.ROOK] * (len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))) +
        PIECE_VALUES[chess.BISHOP] * (len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))) +
        PIECE_VALUES[chess.KNIGHT] * (len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)))
    )
    return min(1.0, max(0.0, phase_score / max_phase))

def bishop_pair(board: chess.Board) -> int:
    bonus = 0
    for color in [chess.WHITE, chess.BLACK]:
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            bonus += 30 if color == chess.WHITE else -30
    return bonus

def central_control(board: chess.Board) -> int:
    from .bitboard import CENTER_MASK, EXT_CENTER_MASK, count_bits
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        attacks = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for sq in board.pieces(piece_type, color):
                attacks |= int(board.attacks(sq))
        central = count_bits(attacks & CENTER_MASK)
        extended = count_bits(attacks & EXT_CENTER_MASK)
        bonus = 10 * central + 3 * extended
        score += bonus if color == chess.WHITE else -bonus
    return score

def space_advantage(board: chess.Board) -> int:
    # Count pawns beyond 4th (white) or before 5th (black)
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        for sq in pawns:
            rank = chess.square_rank(sq)
            if (color == chess.WHITE and rank >= 4) or (color == chess.BLACK and rank <= 3):
                score += 8 if color == chess.WHITE else -8
    return score

def is_valid_square(sq):
    return 0 <= sq < 64

def outpost_bonus(board: chess.Board) -> int:
    # Bonus for knights/bishops on outposts (supported by pawn, cannot be attacked by enemy pawn)
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        opp = not color
        pawns = board.pieces(chess.PAWN, color)
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for sq in board.pieces(piece_type, color):
                # Supported by pawn
                if color == chess.WHITE:
                    support = (is_valid_square(sq - 9) and sq - 9 in pawns) or (is_valid_square(sq - 7) and sq - 7 in pawns)
                    safe = not ((is_valid_square(sq + 7) and sq + 7 in board.pieces(chess.PAWN, opp)) or (is_valid_square(sq + 9) and sq + 9 in board.pieces(chess.PAWN, opp)))
                else:
                    support = (is_valid_square(sq + 9) and sq + 9 in pawns) or (is_valid_square(sq + 7) and sq + 7 in pawns)
                    safe = not ((is_valid_square(sq - 7) and sq - 7 in board.pieces(chess.PAWN, opp)) or (is_valid_square(sq - 9) and sq - 9 in board.pieces(chess.PAWN, opp)))
                if support and safe:
                    score += 18 if color == chess.WHITE else -18
    return score

def tactical_features(board: chess.Board) -> int:
    # Hanging pieces: attacked and not defended
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        opp = not color
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                if board.is_attacked_by(opp, sq) and not board.is_attacked_by(color, sq):
                    score -= PIECE_VALUES.get(piece.piece_type, 0) // 8 if color == chess.WHITE else -PIECE_VALUES.get(piece.piece_type, 0) // 8
    # TODO: Pins, forks, mate threats (bitboard-based, fast)
    return score

def evaluate(board: chess.Board) -> int:
    """Tapered, multi-component evaluation for Search Engine V3."""
    w = tapered_weight(board)
    # Midgame and endgame features
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
    # Draw contempt: prefer not to draw
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        score += 20 if board.turn == chess.WHITE else -20
    return score
