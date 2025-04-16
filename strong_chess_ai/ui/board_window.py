import tkinter as tk
import chess
from .view import ChessBoardCanvas, SidePanel
from .controller import ChessController
import argparse

class BoardWindow(tk.Tk):
    def __init__(self, threads=None):
        super().__init__()
        self.title("Chess AI Board - Search Engine V3")
        self.geometry("1040x640")
        self.resizable(False, False)
        # Initialize the controller with parallel thread support if provided.
        self.controller = ChessController(self, threads=threads)
        # The board canvas displays the current chessboard.
        self.board = ChessBoardCanvas(self, self.controller)
        self.board.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        # The side panel displays move history and extra info.
        self.side_panel = SidePanel(self, self.controller)
        self.side_panel.pack(side=tk.RIGHT, fill=tk.Y)
        # A status label to display game and search state.
        self.status = tk.Label(self, text="", font=("Arial", 16))
        self.status.pack(fill=tk.X)
        # Begin periodic status updates (every 500ms)
        self.update_status()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        """Cleanly close the board window and underlying components."""
        try:
            self.board.destroy()
        except Exception:
            pass
        self.destroy()

    def update_status(self):
        """
        Update the status bar with game state messages and, if available,
        search statistics (such as search depth, nodes, and principal variation).
        This function schedules itself to run every 500ms.
        """
        board = self.controller.state.board
        status_text = ""

        if board.is_checkmate():
            winner = "White" if board.turn == chess.WHITE else "Black"
            status_text = f"Mate – {winner} wins"
        elif board.is_stalemate():
            status_text = "Stalemate"
        elif board.is_check():
            status_text = "Check"
        else:
            # If the AI is thinking, display current search stats.
            # It is assumed that ChessController maintains a property 'search_stats'
            # and a method is_ai_thinking() that returns True while the AI is searching.
            if self.controller.is_ai_thinking():
                stats = self.controller.search_stats
                if stats and stats.pv:
                    pv_str = " ".join(move.uci() for move in stats.pv)
                    status_text = f"Thinking... Depth: {stats.depth} | Nodes: {stats.nodes} | PV: {pv_str}"
                else:
                    status_text = "Thinking..."
            else:
                status_text = ""
        self.status.config(text=status_text)
        self.after(500, self.update_status)

def main():
    parser = argparse.ArgumentParser(description="Chess AI Board - Search Engine V3")
    parser.add_argument('--threads', type=int, default=None,
                        help='Number of threads for parallel root search (default: cpu_count)')
    args = parser.parse_args()
    app = BoardWindow(threads=args.threads)
    app.mainloop()

if __name__ == "__main__":
    main()
