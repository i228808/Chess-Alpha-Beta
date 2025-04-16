import chess
import pytest
from strong_chess_ai.core.board import GameState

def test_legal_moves_initial():
    gs = GameState()
    moves = gs.legal_moves()
    assert chess.Move.from_uci('e2e4') in moves
    assert chess.Move.from_uci('e7e5') not in moves

def test_push_and_pop():
    gs = GameState()
    move = chess.Move.from_uci('e2e4')
    gs.push(move)
    assert gs.move_history[-1] == move
    gs.pop()
    assert len(gs.move_history) == 0

def test_checkmate_detection():
    gs = GameState()
    # Fool's mate
    gs.push(chess.Move.from_uci('f2f3'))
    gs.push(chess.Move.from_uci('e7e5'))
    gs.push(chess.Move.from_uci('g2g4'))
    gs.push(chess.Move.from_uci('d8h4'))
    assert gs.is_terminal()
    assert gs.result_str() == '0-1'

def test_stalemate_detection():
    gs = GameState()
    # Set up a known stalemate position
    gs.board.set_fen('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1')
    gs.zobrist = gs.board.zobrist_hash()
    assert gs.is_terminal()
    assert gs.result_str() == '1/2-1/2'
