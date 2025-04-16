import chess
import timeit
from strong_chess_ai.core.eval import pawn_structure

def test_pawn_structure_perf():
    boards = []
    for _ in range(100):
        board = chess.Board()
        # Make 10 random pawn moves
        for _ in range(10):
            moves = [m for m in board.legal_moves if board.piece_type_at(m.from_square) == chess.PAWN]
            if not moves:
                break
            move = moves[0]
            board.push(move)
        boards.append(board.copy())
    # Time pawn_structure on all boards
    t = timeit.timeit(lambda: [pawn_structure(b) for b in boards], number=10)
    avg_us = (t / (10 * len(boards))) * 1e6
    print(f"Average pawn_structure time: {avg_us:.2f} us")
    assert avg_us < 50, f"pawn_structure() too slow: {avg_us:.2f} us"
