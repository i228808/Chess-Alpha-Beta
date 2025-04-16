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

def test_krb_vs_k_white_mate():
    # White to mate in 1: Kd6, Bc6, Rh8, Black Kb8
    board = chess.Board()
    board.set_fen('1k6/8/2BK4/8/8/8/8/7R w - - 0 1')
    score = ce.evaluate(board)
    assert score >= 100000  # INF for white

def test_krb_vs_k_black_mate():
    # Black to mate in 1: Kd3, Bc3, Rh1, White Kb1
    board = chess.Board()
    board.set_fen('1K6/8/8/8/8/2bk4/8/7r b - - 0 1')
    score = ce.evaluate(board)
    assert score <= -100000  # -INF for black
