import chess
from strong_chess_ai.core import eval as ce

def test_kpk_win():
    # White wins: King supports pawn
    board = chess.Board()
    board.set_fen('8/8/8/8/8/4k3/3P4/3K4 w - - 0 1')
    score = ce.evaluate(board)
    assert score > 0

def test_kpk_draw():
    # Draw: Wrong king position
    board = chess.Board()
    board.set_fen('8/8/8/8/8/3k4/3P4/3K4 w - - 0 1')
    score = ce.evaluate(board)
    assert abs(score) < 50

def test_k_vs_k():
    board = chess.Board()
    board.set_fen('8/8/8/8/8/8/8/3K1k2 w - - 0 1')
    score = ce.evaluate(board)
    assert score == 0

def test_kn_vs_k():
    board = chess.Board()
    board.set_fen('8/8/8/8/8/8/8/2N2k2 w - - 0 1')
    score = ce.evaluate(board)
    assert score == 0

def test_kb_vs_k():
    board = chess.Board()
    board.set_fen('8/8/8/8/8/8/8/2B2k2 w - - 0 1')
    score = ce.evaluate(board)
    assert score == 0

def test_knn_vs_k():
    board = chess.Board()
    board.set_fen('8/8/8/8/8/8/8/1NN3k2 w - - 0 1')
    score = ce.evaluate(board)
    assert score == 0
