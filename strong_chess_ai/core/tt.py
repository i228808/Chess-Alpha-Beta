"""
Transposition Table (TT) with flag handling.
"""
from collections import namedtuple

TTEntry = namedtuple('TTEntry', ['key', 'depth', 'score', 'flag', 'best_move'])

class TranspositionTable:
    def __init__(self, size=2**20):
        self.size = size
        self.table = dict()
    def get(self, key):
        return self.table.get(key, None)
    def store(self, key, depth, score, flag, best_move):
        self.table[key] = TTEntry(key, depth, score, flag, best_move)
    def clear(self):
        self.table.clear()
