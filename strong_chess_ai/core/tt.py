from collections import namedtuple
from typing import Optional, Any
import random

TTEntry = namedtuple('TTEntry', ['key', 'depth', 'score', 'flag', 'best_move'])

class TranspositionTable:
    def __init__(self, size: int = 2**20) -> None:
        """
        Initialize the transposition table with a given maximum size.

        Args:
            size (int): Maximum number of entries allowed in the table.
        """
        self.size: int = size
        self.table: dict[int, TTEntry] = {}

    def get(self, key: int) -> Optional[TTEntry]:
        """
        Retrieve the transposition table entry for the given key.

        Args:
            key (int): The unique hash key for a board position.

        Returns:
            Optional[TTEntry]: The stored entry or None if not found.
        """
        return self.table.get(key, None)

    def store(self, key: int, depth: int, score: int, flag: str, best_move: Optional[Any]) -> None:
        """
        Store or update an entry in the transposition table.

        If an entry for the key already exists and its recorded depth is greater or equal,
        the new entry will not replace it. Otherwise, the new entry is stored.

        If the table exceeds the maximum allowed entries, one entry is removed at random.

        Args:
            key (int): Unique hash key for the board position.
            depth (int): Depth at which the score was computed.
            score (int): Evaluated score of the position.
            flag (str): A flag indicating if the score is 'EXACT', a lower bound ('LOWER'),
                        or an upper bound ('UPPER').
            best_move (Optional[Any]): The best move from this position, if any.
        """
        existing_entry = self.table.get(key)
        if existing_entry is not None and existing_entry.depth >= depth:
            # Do not replace if the existing entry is at least as deep.
            return

        # If table exceeds capacity, remove a random entry.
        if len(self.table) >= self.size:
            random_key = random.choice(list(self.table.keys()))
            del self.table[random_key]

        self.table[key] = TTEntry(key, depth, score, flag, best_move)

    def clear(self) -> None:
        """
        Clear all entries from the transposition table.
        """
        self.table.clear()

    def __contains__(self, key: int) -> bool:
        """
        Check if the given key exists in the transposition table.

        Args:
            key (int): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self.table
