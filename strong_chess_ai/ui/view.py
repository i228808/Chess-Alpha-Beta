# GUI code removed for redesign

class ChessBoardCanvas(tk.Canvas):
    def clear_selection_and_redraw(self):
        """Clears selection and redraws the board."""
        self.selected_square = None
        self.legal_squares = []
        self.redraw()

    def __init__(self, master, controller, **kwargs):
        super().__init__(master, width=BOARD_SIZE, height=BOARD_SIZE, **kwargs)
        self.controller = controller
        self.selected_square = None
        self.legal_squares = []
        self.fps = 60
        self.after_id = None
        self._destroyed = False
        self.bind("<Button-1>", self.on_click)
        self.redraw()

    def on_click(self, event):
        """Handles the click events for selecting and moving pieces."""
        file = event.x // SQUARE_SIZE
        rank = 7 - (event.y // SQUARE_SIZE)
        sq = chess.square(file, rank)
        board = self.controller.state.board
        legal_moves = list(board.legal_moves)
        legal_from_squares = set(move.from_square for move in legal_moves)
        piece = board.piece_at(sq)

        if not legal_moves:
            if board.is_checkmate():
                winner = "Black" if board.turn == chess.WHITE else "White"
                result = f"Checkmate! {winner} wins."
            elif board.is_stalemate():
                result = "Stalemate! Draw."
            else:
                result = "Game over."
            messagebox.showinfo("Game Over", result)
            return

        if self.selected_square is None:
            if piece and piece.color == board.turn and sq in legal_from_squares:
                self.selected_square = sq
                self.legal_squares = [move.to_square for move in legal_moves if move.from_square == sq]
        else:
            move = chess.Move(self.selected_square, sq)
            if move in legal_moves:
                self.controller.make_human_move(move)
            self.selected_square = None
            self.legal_squares = []

        self.redraw()

    def redraw(self):
        """Redraws the chessboard."""
        if self._destroyed or not self.winfo_exists():
            return
        self.delete("all")
        for rank in range(8):
            for file in range(8):
                x0 = file * SQUARE_SIZE
                y0 = (7 - rank) * SQUARE_SIZE
                color = LIGHT if (rank + file) % 2 == 0 else DARK
                if self.selected_square == chess.square(file, rank):
                    color = "#88f"
                elif chess.square(file, rank) in self.legal_squares:
                    color = "#8f8"
                self.create_rectangle(x0, y0, x0 + SQUARE_SIZE, y0 + SQUARE_SIZE, fill=color, outline="")
        
        for sq in chess.SQUARES:
            piece = self.controller.state.board.piece_at(sq)
            if piece:
                color = "w" if piece.color == chess.WHITE else "b"
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                x = file * SQUARE_SIZE
                y = (7 - rank) * SQUARE_SIZE
                uni = UNICODE_PIECES[piece.symbol()]
                self.create_text(x + SQUARE_SIZE//2, y + SQUARE_SIZE//2, text=uni, font=("Arial", SQUARE_SIZE//2 + 10), fill="black" if color=="b" else "white")

        if not self._destroyed and self.winfo_exists():
            self.after_id = self.after(int(1000/self.fps), self.redraw)


class SidePanel(tk.Frame):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, width=SIDE_PANEL_WIDTH, **kwargs)
        self.controller = controller
        self.move_list = tk.Listbox(self, font=("Arial", 12), width=18)
        self.move_list.pack(fill=tk.BOTH, expand=True)
        self.undo_btn = tk.Button(self, text="Undo", command=controller.undo)
        self.undo_btn.pack(fill=tk.X)
        self.flip_btn = tk.Button(self, text="Flip Board", command=controller.flip_board)
        self.flip_btn.pack(fill=tk.X)
        self.depth_label = tk.Label(self, text=f"Depth: {controller.depth}")
        self.depth_label.pack()
        self.depth_up = tk.Button(self, text="Depth +", command=controller.increase_depth)
        self.depth_up.pack(fill=tk.X)
        self.depth_down = tk.Button(self, text="Depth -", command=controller.decrease_depth)
        self.depth_down.pack(fill=tk.X)

    def update_moves(self, moves):
        self.move_list.delete(0, tk.END)
        for move in moves:
            self.move_list.insert(tk.END, move)

    def update_depth(self, depth):
        self.depth_label.config(text=f"Depth: {depth}")


class BoardWindow(tk.Tk):
    def __init__(self, threads=None):
        super().__init__()
        self.title("Chess AI Board - Search Engine V3")
        self.geometry("1040x640")
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
        self.controller.ai_thread.join()
        self.destroy()

    def update_status(self):
        """Updates the status label with the current game state."""
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
