import tkinter as tk
from .view import ChessBoardCanvas, SidePanel
from .controller import ChessController

class BoardWindow(tk.Tk):
    def __init__(self, threads=None):
        super().__init__()
        self.title("Chess AI Board")
        self.geometry(f"1040x640")
        self.resizable(False, False)
        self.controller = ChessController(self, threads=threads)
        self.board = ChessBoardCanvas(self, self.controller)
        self.board.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.side_panel = SidePanel(self, self.controller)
        self.side_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.status = tk.Label(self, text="", font=("Arial", 16))
        self.status.pack(fill=tk.X)
        self.update_status()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        try:
            self.board.destroy()
        except Exception:
            pass
        self.destroy()

    def update_status(self):
        board = self.controller.state.board
        if board.is_checkmate():
            winner = "White" if board.turn == 0 else "Black"
            self.status.config(text=f"Mate – {winner} wins")
        elif board.is_stalemate():
            self.status.config(text="Stalemate")
        elif board.is_check():
            self.status.config(text="Check")
        else:
            self.status.config(text="")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Chess AI Board")
    parser.add_argument('--threads', type=int, default=None, help='Number of threads for parallel root search (default: cpu_count)')
    args = parser.parse_args()
    app = BoardWindow(threads=args.threads)
    app.mainloop()

if __name__ == "__main__":
    main()
