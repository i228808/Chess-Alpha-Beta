"""
Book module for strong_chess_ai: builds and queries a Polyglot-style opening book.
"""
import chess
import chess.pgn
import chess.polyglot
import random
import struct
from collections import defaultdict
from typing import Optional

BOOK_FILE = "book.bin"

class BookEntry:
    def __init__(self, key: int, move: chess.Move, count: int):
        self.key = key
        self.move = move
        self.count = count

    def to_bytes(self) -> bytes:
        # Polyglot format: key (8), move (2), weight (2), learn (4)
        move16 = chess.polyglot.encode_move(self.move)
        return struct.pack(
            ">QHHI", self.key, move16, self.count, 0
        )

    @staticmethod
    def from_bytes(data: bytes) -> "BookEntry":
        key, move16, count, _ = struct.unpack(">QHHI", data)
        return BookEntry(key, chess.polyglot.decode_move(move16), count)


def build_polyglot(pgn_path: str, max_moves: int = 12, output_path: str = BOOK_FILE):
    """Build a Polyglot-style book from a PGN file up to max_moves (plies)."""
    book = defaultdict(lambda: defaultdict(int))  # key -> move -> count
    with open(pgn_path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = chess.Board()
            for i, move in enumerate(game.mainline_moves()):
                if i >= max_moves:
                    break
                key = chess.polyglot.zobrist_hash(board)
                book[key][move] += 1
                board.push(move)
    # Write to .bin
    with open(output_path, "wb") as out:
        for key, moves in book.items():
            for move, count in moves.items():
                entry = BookEntry(key, move, count)
                out.write(entry.to_bytes())


def lookup(state) -> Optional[chess.Move]:
    """Return a random weighted book move for the current position, or None."""
    try:
        with open(BOOK_FILE, "rb") as f:
            entries = []
            key = state.zobrist
            while True:
                data = f.read(16)
                if not data or len(data) < 16:
                    break
                entry = BookEntry.from_bytes(data)
                if entry.key == key:
                    entries.append((entry.move, entry.count))
        if entries:
            moves, weights = zip(*entries)
            return random.choices(moves, weights=weights)[0]
        return None
    except FileNotFoundError:
        return None
