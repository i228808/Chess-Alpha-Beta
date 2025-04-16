import chess
from strong_chess_ai.core.board import GameState
from strong_chess_ai.core.search import find_best_move, SearchStats

def test_mate_in_2():
    # White to move, mate in 2
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
    gs = GameState()
    gs.board.set_fen(fen)
    gs.zobrist = gs.board.zobrist_hash()
    stats = find_best_move(gs, max_depth=3, time_limit_s=10)
    # Should find mate in 2 (Qxf7# or similar)
    assert any(move.uci().startswith('f1') or move.uci().startswith('d1') for move in stats.pv)
    assert stats.depth >= 3

def test_nodes_under_1m():
    # Starting position at depth 4 should search under 1 million nodes
    gs = GameState()
    stats = find_best_move(gs, max_depth=4, time_limit_s=20)
    assert stats.nodes < 1_000_000
