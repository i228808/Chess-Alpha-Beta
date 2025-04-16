import chess
from strong_chess_ai.core import eval as ce

def test_isolated_pawn():
    board = chess.Board()
    board.set_fen('8/8/8/8/8/8/8/8 w - - 0 1')
    board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.A2, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.H2, chess.Piece(chess.PAWN, chess.WHITE))
    assert ce._isolated_mask(4) == 0b10000001
    # e4 is isolated, a2 and h2 are also isolated
    assert ce.pawn_structure(board) < 0

def test_doubled_pawns():
    board = chess.Board()
    board.set_fen('8/8/8/8/8/8/8/8 w - - 0 1')
    board.set_piece_at(chess.D2, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.D3, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.D4, chess.Piece(chess.PAWN, chess.WHITE))
    assert ce._doubled_count(3) == 2
    assert ce.pawn_structure(board) < 0

def test_passed_pawn_bonus():
    board = chess.Board()
    board.set_fen('8/8/4P3/8/8/8/8/8 w - - 0 1')
    # White pawn on e6 (rank 6)
    assert ce.pawn_structure(board) > 0
    board.set_fen('8/8/8/4P3/8/8/8/8 w - - 0 1')
    # White pawn on e5 (rank 5)
    assert ce.pawn_structure(board) > 0
    board.set_fen('8/8/8/8/4P3/8/8/8 w - - 0 1')
    # White pawn on e4 (rank 4) -- no bonus
    assert ce.pawn_structure(board) == 0

def test_black_passed_pawn_bonus():
    board = chess.Board()
    board.set_fen('8/8/8/8/8/8/4p3/8 b - - 0 1')
    # Black pawn on e3 (rank 2 for black)
    assert ce.pawn_structure(board) == 0
    board.set_fen('8/8/8/8/8/4p3/8/8 b - - 0 1')
    # Black pawn on e4 (rank 3 for black)
    assert ce.pawn_structure(board) == 0
    board.set_fen('8/8/8/4p3/8/8/8/8 b - - 0 1')
    # Black pawn on e5 (rank 5 for black, bonus)
    assert ce.pawn_structure(board) < 0
