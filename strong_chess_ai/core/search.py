"""
Search module for strong_chess_ai: implements iterative deepening alpha-beta with PV, TT, killer/history heuristics, and MVV/LVA.
"""
from dataclasses import dataclass
import chess
import time
from typing import Optional, Dict, Tuple, List
from strong_chess_ai.core.board import GameState
from strong_chess_ai.core import eval as ce

DEPTH_MAX = 32
INF = 100000

@dataclass
class TTEntry:
    key: int
    depth: int
    score: int
    flag: str  # 'EXACT', 'LOWER', 'UPPER'
    best_move: Optional[chess.Move]

@dataclass
class SearchStats:
    nodes: int = 0
    cutoffs: int = 0
    depth: int = 0
    pv: List[chess.Move] = None

# Transposition table
tt: Dict[int, TTEntry] = {}
# Killer moves: DEPTH_MAX x 2
killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(DEPTH_MAX)]
# History heuristic: (from,to) -> int
history_heuristic: Dict[Tuple[int, int], int] = {}

# MVV/LVA ordering for captures
def mvv_lva(move: chess.Move, board: chess.Board) -> int:
    if not board.is_capture(move):
        return 0
    victim = board.piece_type_at(move.to_square)
    if victim is None:
        # en passant
        victim = chess.PAWN
    attacker = board.piece_type_at(move.from_square)
    return 10 * ce.PIECE_VALUES[victim] - ce.PIECE_VALUES.get(attacker, 0)

def order_moves(state: GameState, moves: List[chess.Move], depth: int, pv_move: Optional[chess.Move]) -> List[chess.Move]:
    def score(move):
        if pv_move and move == pv_move:
            return 1000000
        if state.board.is_capture(move):
            return 500000 + mvv_lva(move, state.board)
        # Killer moves
        killers = killer_moves[depth] if depth < DEPTH_MAX else []
        if move in killers:
            return 200000
        # History heuristic
        return history_heuristic.get((move.from_square, move.to_square), 0)
    return sorted(moves, key=score, reverse=True)

def alphabeta(state: GameState, depth: int, alpha: int, beta: int, ply: int = 0, pv: Optional[List[chess.Move]] = None, root: bool = False) -> Tuple[int, Optional[chess.Move]]:
    stats = alphabeta.stats
    stats.nodes += 1
    key = state.zobrist
    if key in tt:
        entry = tt[key]
        if entry.depth >= depth:
            if entry.flag == 'EXACT':
                return entry.score, entry.best_move
            elif entry.flag == 'LOWER' and entry.score > alpha:
                alpha = entry.score
            elif entry.flag == 'UPPER' and entry.score < beta:
                beta = entry.score
            if alpha >= beta:
                stats.cutoffs += 1
                return entry.score, entry.best_move
    if depth == 0 or state.is_terminal():
        return quiesce(state, alpha, beta, stats), None
    best_score = -INF
    best_move = None
    legal_moves = list(state.legal_moves())
    pv_move = pv[ply] if pv and ply < len(pv) else None
    ordered_moves = order_moves(state, legal_moves, ply, pv_move)
    for move in ordered_moves:
        state.push(move)
        score, _ = alphabeta(state, depth - 1, -beta, -alpha, ply + 1, pv, False)
        score = -score
        state.pop()
        if score > best_score:
            best_score = score
            best_move = move
            if root and pv is not None:
                if len(pv) <= ply:
                    pv.append(move)
                else:
                    pv[ply] = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            stats.cutoffs += 1
            # Killer move
            if move not in killer_moves[ply]:
                killer_moves[ply][1] = killer_moves[ply][0]
                killer_moves[ply][0] = move
            # History heuristic
            if not state.board.is_capture(move):
                history_heuristic[(move.from_square, move.to_square)] = history_heuristic.get((move.from_square, move.to_square), 0) + depth * depth
            break
    # Store TT
    flag = 'EXACT' if best_score > alpha and best_score < beta else ('LOWER' if best_score >= beta else 'UPPER')
    tt[key] = TTEntry(key, depth, best_score, flag, best_move)
    return best_score, best_move
alphabeta.stats = SearchStats()

def quiesce(state: GameState, alpha: int, beta: int, stats: Optional[SearchStats] = None) -> int:
    if stats: stats.nodes += 1
    stand_pat = ce.evaluate(state.board)
    if stand_pat >= beta:
        if stats: stats.cutoffs += 1
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    for move in state.legal_moves():
        if state.board.is_capture(move) or state.board.gives_check(move):
            state.push(move)
            score = -quiesce(state, -beta, -alpha, stats)
            state.pop()
            if score >= beta:
                if stats: stats.cutoffs += 1
                return beta
            if score > alpha:
                alpha = score
    return alpha

def find_best_move(state: GameState, max_depth: int = 5, time_limit_s: float = 5.0) -> SearchStats:
    """Iterative deepening with PV move ordering and TT, returns SearchStats."""
    stats = SearchStats()
    pv = []
    start_time = time.time()
    best_move = None
    for depth in range(1, max_depth + 1):
        alphabeta.stats = stats
        pv = []
        score, move = alphabeta(state, depth, -INF, INF, 0, pv, True)
        stats.depth = depth
        best_move = move
        if time.time() - start_time > time_limit_s:
            break
    stats.pv = pv[:stats.depth]
    return stats
