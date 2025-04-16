import tkinter as tk
from .resources import SQUARE_SIZE, BOARD_SIZE, SIDE_PANEL_WIDTH, LIGHT, DARK, PIECE_IMAGES, UNICODE_PIECES
import chess

class ChessBoardCanvas(tk.Canvas):
    def __init__(self, master, controller, **kwargs):
        super().__init__(master, width=BOARD_SIZE, height=BOARD_SIZE, **kwargs)
        self.controller = controller
        self.bind("<ButtonPress-1>", self.on_drag_start)
        self.bind("<B1-Motion>", self.on_drag_motion)
        self.bind("<ButtonRelease-1>", self.on_drag_release)
        self.drag_data = {"from": None, "piece": None, "img": None}
        self.fps = 60
        self.after_id = None
        self._destroyed = False
        self.redraw()

    def destroy(self):
        self._destroyed = True
        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
        super().destroy()

    def redraw(self):
        if self._destroyed or not self.winfo_exists():
            return
        self.delete("all")
        for rank in range(8):
            for file in range(8):
                x0 = file * SQUARE_SIZE
                y0 = (7 - rank) * SQUARE_SIZE
                color = LIGHT if (rank + file) % 2 == 0 else DARK
                self.create_rectangle(x0, y0, x0 + SQUARE_SIZE, y0 + SQUARE_SIZE, fill=color, outline="")
        # Draw pieces
        for sq in chess.SQUARES:
            piece = self.controller.state.board.piece_at(sq)
            if piece:
                color = "w" if piece.color == chess.WHITE else "b"
                name = piece.symbol().upper()
                img = PIECE_IMAGES.get(color + name)
                file, rank = chess.square_file(sq), chess.square_rank(sq)
                x = file * SQUARE_SIZE
                y = (7 - rank) * SQUARE_SIZE
                if img:
                    self.create_image(x, y, anchor=tk.NW, image=img)
                else:
                    uni = UNICODE_PIECES[piece.symbol()]
                    self.create_text(x + SQUARE_SIZE//2, y + SQUARE_SIZE//2, text=uni, font=("Arial", SQUARE_SIZE//2 + 10), fill="black" if color=="b" else "white")
        # Drag highlight
        if self.drag_data["from"] is not None:
            f, r = chess.square_file(self.drag_data["from"]), chess.square_rank(self.drag_data["from"])
            x0 = f * SQUARE_SIZE
            y0 = (7 - r) * SQUARE_SIZE
            self.create_rectangle(x0, y0, x0 + SQUARE_SIZE, y0 + SQUARE_SIZE, outline="#00f", width=3)
        # FPS cap
        if not self._destroyed and self.winfo_exists():
            self.after_id = self.after(int(1000/self.fps), self.redraw)

    def on_drag_start(self, event):
        file = event.x // SQUARE_SIZE
        rank = 7 - (event.y // SQUARE_SIZE)
        sq = chess.square(file, rank)
        piece = self.controller.state.board.piece_at(sq)
        if piece and piece.color == self.controller.state.board.turn:
            self.drag_data["from"] = sq
            self.drag_data["piece"] = piece

    def on_drag_motion(self, event):
        pass  # Could add ghost piece, but not needed for basic DnD

    def on_drag_release(self, event):
        if self.drag_data["from"] is None:
            return
        file = event.x // SQUARE_SIZE
        rank = 7 - (event.y // SQUARE_SIZE)
        to_sq = chess.square(file, rank)
        move = chess.Move(self.drag_data["from"], to_sq)
        if move in self.controller.state.board.legal_moves:
            self.controller.make_human_move(move)
        self.drag_data = {"from": None, "piece": None, "img": None}

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
        for san in moves:
            self.move_list.insert(tk.END, san)

    def update_depth(self, depth):
        self.depth_label.config(text=f"Depth: {depth}")
