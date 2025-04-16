import threading
import chess
from .resources import play_sound, SOUND_MOVE, SOUND_CAPTURE
from strong_chess_ai.core.board import GameState
from strong_chess_ai.core.search import find_best_move

class ChessController:
    def __init__(self, view, threads=None):
        self.view = view
        self.state = GameState()
        self.depth = 5
        self.threads = threads
        self.ai_thread = None
        self.flipped = False
        self.move_history = []
        self.lock = threading.Lock()

    def make_human_move(self, move):
        with self.lock:
            san = self.state.board.san(move)
            capture = self.state.board.is_capture(move)
            self.state.push(move)
            play_sound(SOUND_CAPTURE if capture else SOUND_MOVE)
            self.move_history.append(san)
            self.view.side_panel.update_moves(self.move_history)
            self.view.board.redraw()
            self.view.update_status()
            if not self.state.is_terminal():
                self.view.after(100, self.spawn_ai)

    def spawn_ai(self):
        def ai_move():
            result = find_best_move(self.state, max_depth=self.depth, time_limit_s=3.0, threads=self.threads)
            move = result.pv[0] if result.pv else None
            if move:
                with self.lock:
                    san = self.state.board.san(move)
                    capture = self.state.board.is_capture(move)
                    self.state.push(move)
                    play_sound(SOUND_CAPTURE if capture else SOUND_MOVE)
                    self.move_history.append(san)
                    self.view.side_panel.update_moves(self.move_history)
                    self.view.board.redraw()
                    self.view.update_status()
        self.ai_thread = threading.Thread(target=ai_move, daemon=True)
        self.ai_thread.start()

    def undo(self):
        with self.lock:
            if len(self.state.move_history) >= 2:
                self.state.pop()
                self.state.pop()
                self.move_history = self.move_history[:-2]
                self.view.side_panel.update_moves(self.move_history)
                self.view.board.redraw()
                self.view.update_status()

    def flip_board(self):
        self.flipped = not self.flipped
        # Not implemented: would require board redraw logic

    def increase_depth(self):
        self.depth += 1
        self.view.side_panel.update_depth(self.depth)

    def decrease_depth(self):
        if self.depth > 1:
            self.depth -= 1
            self.view.side_panel.update_depth(self.depth)
