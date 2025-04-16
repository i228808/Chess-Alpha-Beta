import threading
import chess
from .resources import play_sound, SOUND_MOVE, SOUND_CAPTURE
from strong_chess_ai.core.board import GameState
from strong_chess_ai.core.search import find_best_move
import logging


class ChessController:
    def __init__(self, view, threads=None):
        self.view = view
        self.state = GameState()
        self.depth = 5
        self.threads = threads
        self.ai_thread = None
        self.flipped = False
        self.move_history = []
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.search_stats = None  # Holds latest SearchStats from the search module

    def is_ai_thinking(self) -> bool:
        """
        Returns True if the AI thread is currently active.
        """
        return self.ai_thread is not None and self.ai_thread.is_alive()

    def make_human_move(self, move):
        with self.lock:
            # Ensure only one move is made at a time and no race conditions
            move_number = self.state.board.fullmove_number
            if move not in self.state.legal_moves():
                logging.error(f"Attempted illegal move: {move}")
                return
            capture = self.state.board.is_capture(move)
            san = self.state.board.san(move)
            self.state.push(move)
            play_sound(SOUND_CAPTURE if capture else SOUND_MOVE)
            self.move_history.append(san)

        # Update the UI to reflect the new move
        self.view.after(0, self.view.side_panel.update_moves, self.move_history)
        self.view.after(0, self.view.board.redraw)
        self.view.after(0, self.view.update_status)

        # If the game is not over, trigger the AI to make its move
        if not self.state.is_terminal():
            self.view.after(100, self.spawn_ai)

    def spawn_ai(self):
        """
        Triggers the AI to make its move in a separate thread.
        """
        def ai_move():
            try:
                # Run the AI logic in a separate thread to avoid blocking the UI
                result = find_best_move(self.state, max_depth=self.depth, time_limit_s=3.0, threads=self.threads)
                if result.pv:
                    move = result.pv[0]
                    with self.lock:
                        capture = self.state.board.is_capture(move)
                        san = self.state.board.san(move)
                        self.state.push(move)
                        play_sound(SOUND_CAPTURE if capture else SOUND_MOVE)
                        self.move_history.append(san)
                        self.search_stats = result

                    # UI updates
                    self.view.after(0, self.view.side_panel.update_moves, self.move_history)
                    self.view.after(0, self.view.board.redraw)
                    self.view.after(0, self.view.update_status)

                    # Game over check
                    if self.state.is_terminal():
                        self.view.after(0, lambda: self.view.show_game_over("Game Over"))
            except Exception as e:
                logging.error(f"Error in AI move: {e}")

        # Only allow one AI thread at a time
        if self.ai_thread is not None and self.ai_thread.is_alive():
            logging.warning("AI is already thinking...")
            return

        # Create and start AI thread
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
        self.view.board.redraw()

    def increase_depth(self):
        self.depth += 1
        self.view.side_panel.update_depth(self.depth)

    def decrease_depth(self):
        if self.depth > 1:
            self.depth -= 1
            self.view.side_panel.update_depth(self.depth)
