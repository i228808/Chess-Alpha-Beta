"""
Book module for strong_chess_ai: builds and queries a Polyglot-style opening book.

This module provides functions to build an opening book from PGN games
(up to a given ply depth) and to lookup weighted moves from the book
using a Polyglot-style format.
"""

import chess
import chess.pgn
import chess.polyglot
import random
import struct
from collections import defaultdict
from typing import Optional, Tuple, List

BOOK_FILE: str = "book.bin"

class BookEntry:
    """
    Represents a single entry in the Polyglot-style opening book.
    
    Polyglot format is defined as:
        - key: 8 bytes (unsigned long long)
        - move: 2 bytes (encoded move)
        - weight: 2 bytes (unsigned short) representing the move frequency/count
        - learn: 4 bytes (unsigned int), usually set to 0 (unused)
    """
    def __init__(self, key: int, move: chess.Move, count: int) -> None:
        self.key: int = key
        self.move: chess.Move = move
        self.count: int = count

    def to_bytes(self) -> bytes:
        """
        Serialize the BookEntry to bytes in Polyglot format.
        
        Returns:
            bytes: The byte representation of this entry.
        """
        # Encode the move to 16-bit representation using chess.polyglot
        move16: int = chess.polyglot.encode_move(self.move)
        # Pack the key, move, weight (count), and learn (set to 0)
        return struct.pack(">QHHI", self.key, move16, self.count, 0)

    @staticmethod
    def from_bytes(data: bytes) -> "BookEntry":
        """
        Deserialize bytes into a BookEntry.
        
        Args:
            data (bytes): 16 bytes representing a book entry.
            
        Returns:
            BookEntry: The decoded book entry.
        """
        key, move16, count, _ = struct.unpack(">QHHI", data)
        move: chess.Move = chess.polyglot.decode_move(move16)
        return BookEntry(key, move, count)


def build_polyglot(pgn_path: str, max_moves: int = 12, output_path: str = BOOK_FILE) -> None:
    """
    Build a Polyglot-style book from a PGN file. For each game in the PGN,
    process moves up to max_moves (plies) and record move frequency counts.
    
    This function reads through a PGN file, processes each game, and builds a 
    dictionary of position keys to move frequency counts.
    """
    # Initialize the book as a dictionary: key -> {move: count}
    # This dictionary will store the frequency counts for each move in each position.
    book: defaultdict[int, defaultdict[chess.Move, int]] = defaultdict(lambda: defaultdict(int))
    
    # Open the PGN file for reading
    with open(pgn_path, "r", encoding="utf-8") as pgn_file:
        # Read through the PGN file game by game
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            board = chess.Board()
            for i, move in enumerate(game.mainline_moves()):
                if i >= max_moves:
                    break
                # Compute the Polyglot key for the current position
                key = chess.polyglot.zobrist_hash(board)
                book[key][move] += 1
                board.push(move)
                
    # Write book entries to the binary output file in Polyglot format
    with open(output_path, "wb") as out_file:
        for key, moves in book.items():
            for move, count in moves.items():
                entry = BookEntry(key, move, count)
                out_file.write(entry.to_bytes())


def lookup(state) -> Optional[chess.Move]:
    """
    Lookup a weighted opening move from the Polyglot book for the given game state.
    
    The function reads through the binary book file, collects all entries
    matching the current board's Polyglot key (zobrist hash), and returns
    one move chosen randomly with probability weighted by its frequency.
    
    Args:
        state: A GameState object or any object with an attribute `zobrist`
               representing the current board's Polyglot hash.
               
    Returns:
        Optional[chess.Move]: A book move if one is available, otherwise None.
    """
    try:
        with open(BOOK_FILE, "rb") as f:
            entries: List[Tuple[chess.Move, int]] = []
            key: int = state.zobrist
            # Each book entry is 16 bytes in Polyglot format.
            entry_size: int = 16
            while True:
                data = f.read(entry_size)
                if not data or len(data) < entry_size:
                    break
                entry = BookEntry.from_bytes(data)
                if entry.key == key:
                    entries.append((entry.move, entry.count))
        if entries:
            moves, weights = zip(*entries)
            # Name: Abdullah Mansoor, Roll Number: i228808
# Pick a move at random based on how often it appears in the book.
            return random.choices(list(moves), weights=list(weights))[0]
        return None
    except FileNotFoundError:
        return None
