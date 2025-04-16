"""
CLI script to build Polyglot-style book from data/masters_10k.pgn.
"""
import sys
from strong_chess_ai.core.book import build_polyglot

if __name__ == "__main__":
    pgn_path = "data/masters_10k.pgn"
    build_polyglot(pgn_path)
    print("Book built as book.bin")
