# Modern PyQt6 Chess GUI for strong_chess_ai
# Author: Cascade AI

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget, QSlider, QFileDialog, QMessageBox, QFrame
)
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont, QPixmap, QMouseEvent
from PyQt6.QtCore import Qt, QRect, QSize, QTimer
import chess
import chess.pgn
import os
import threading
from strong_chess_ai.core.search import find_best_move

# --- ChessBoard Widget ---
class ChessBoardWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 480)
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.flipped = False
        self.piece_pixmaps = self.load_piece_pixmaps()
        self.setMouseTracking(True)
        self.drag_start = None
        self.drag_piece = None
        self.drag_pixmap = None
        self.drag_offset = None
        self.move_callback = None

    def load_piece_pixmaps(self):
        # Use Unicode as fallback, but can load images from resources if available
        # For now, just return empty dict
        return {}

    def square_at(self, pos):
        size = min(self.width(), self.height())
        square_size = size // 8
        x, y = pos.x(), pos.y()
        file = x // square_size
        rank = 7 - (y // square_size)
        if self.flipped:
            file = 7 - file
            rank = 7 - rank
        if 0 <= file < 8 and 0 <= rank < 8:
            return chess.square(file, rank)
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        size = min(self.width(), self.height())
        square_size = size // 8
        light = QColor(240, 217, 181)
        dark = QColor(181, 136, 99)
        highlight = QColor(100, 200, 255, 128)
        # Draw squares
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, 7 - rank) if not self.flipped else chess.square(7 - file, rank)
                color = light if (rank + file) % 2 == 0 else dark
                painter.fillRect(file * square_size, rank * square_size, square_size, square_size, color)
                if self.selected_square == sq:
                    painter.fillRect(file * square_size, rank * square_size, square_size, square_size, highlight)
                elif sq in self.legal_moves:
                    painter.fillRect(file * square_size, rank * square_size, square_size, square_size, QColor(180,255,180,100))
        # Draw pieces
        font = QFont("Arial", square_size // 2)
        painter.setFont(font)
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece is None:
                continue
            file = chess.square_file(sq)
            rank = 7 - chess.square_rank(sq)
            if self.flipped:
                file = 7 - file
                rank = 7 - rank
            x = file * square_size
            y = rank * square_size
            symbol = piece.symbol()
            uni = {
                'P': '\u2659', 'N': '\u2658', 'B': '\u2657', 'R': '\u2656', 'Q': '\u2655', 'K': '\u2654',
                'p': '\u265F', 'n': '\u265E', 'b': '\u265D', 'r': '\u265C', 'q': '\u265B', 'k': '\u265A',
            }[symbol]
            painter.setPen(QPen(Qt.GlobalColor.black if piece.color == chess.BLACK else Qt.GlobalColor.white))
            painter.drawText(QRect(x, y, square_size, square_size), Qt.AlignmentFlag.AlignCenter, uni)
        # Draw dragged piece
        if self.drag_piece and self.drag_pixmap and self.drag_offset:
            painter.drawPixmap(self.mapFromGlobal(self.cursor().pos()) - self.drag_offset, self.drag_pixmap)

    def mousePressEvent(self, event: QMouseEvent):
        sq = self.square_at(event.position().toPoint())
        if sq is None:
            return
        piece = self.board.piece_at(sq)
        if piece and piece.color == self.board.turn:
            self.selected_square = sq
            self.legal_moves = [move.to_square for move in self.board.legal_moves if move.from_square == sq]
            self.update()
        else:
            self.selected_square = None
            self.legal_moves = []
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.selected_square is None:
            return
        sq = self.square_at(event.position().toPoint())
        if sq is None or sq == self.selected_square:
            self.selected_square = None
            self.legal_moves = []
            self.update()
            return
        move = chess.Move(self.selected_square, sq)
        if move in self.board.legal_moves:
            if self.move_callback:
                self.move_callback(move)
        self.selected_square = None
        self.legal_moves = []
        self.update()

    def set_board(self, board):
        self.board = board
        self.selected_square = None
        self.legal_moves = []
        self.update()

    def set_flipped(self, flipped: bool):
        self.flipped = flipped
        self.update()

# --- Main Window ---
class ChessMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Strong Chess AI - PyQt6 Edition")
        self.setMinimumSize(900, 600)
        self.board_widget = ChessBoardWidget()
        self.move_list = QListWidget()
        self.status_label = QLabel()
        self.flip_btn = QPushButton("Flip Board")
        self.undo_btn = QPushButton("Undo")
        self.redo_btn = QPushButton("Redo")
        self.new_game_btn = QPushButton("New Game")
        self.depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.depth_slider.setMinimum(1)
        self.depth_slider.setMaximum(8)
        self.depth_slider.setValue(4)
        self.depth_slider.setTickInterval(1)
        self.depth_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.depth_label = QLabel("AI Depth: 4")
        self.dark_mode_btn = QPushButton("Dark Mode")
        self.export_pgn_btn = QPushButton("Export PGN")
        self.import_pgn_btn = QPushButton("Import PGN")
        self.hint_btn = QPushButton("Hint")
        self.analysis_label = QLabel()
        self._setup_layout()
        self._connect_signals()
        self._reset_game()

    def _setup_layout(self):
        central = QWidget()
        main_layout = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        left.addWidget(self.board_widget, stretch=1)
        left.addWidget(self.status_label)
        left.addWidget(self.analysis_label)
        right.addWidget(QLabel("Move History:"))
        right.addWidget(self.move_list)
        right.addWidget(self.flip_btn)
        right.addWidget(self.undo_btn)
        right.addWidget(self.redo_btn)
        right.addWidget(self.new_game_btn)
        right.addWidget(self.depth_label)
        right.addWidget(self.depth_slider)
        right.addWidget(self.dark_mode_btn)
        right.addWidget(self.export_pgn_btn)
        right.addWidget(self.import_pgn_btn)
        right.addWidget(self.hint_btn)
        main_layout.addLayout(left, stretch=3)
        main_layout.addLayout(right, stretch=1)
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _connect_signals(self):
        self.board_widget.move_callback = self._on_human_move
        self.flip_btn.clicked.connect(self._on_flip)
        self.undo_btn.clicked.connect(self._on_undo)
        self.redo_btn.clicked.connect(self._on_redo)
        self.new_game_btn.clicked.connect(self._on_new_game)
        self.depth_slider.valueChanged.connect(self._on_depth_change)
        self.dark_mode_btn.clicked.connect(self._on_dark_mode)
        self.export_pgn_btn.clicked.connect(self._on_export_pgn)
        self.import_pgn_btn.clicked.connect(self._on_import_pgn)
        self.hint_btn.clicked.connect(self._on_hint)
        self.move_list.itemClicked.connect(self._on_move_selected)

    def _reset_game(self):
        from strong_chess_ai.core.board import GameState
        self.state = GameState()
        self.board = self.state.board
        self.move_history = []
        self.redo_stack = []
        self.board_widget.set_board(self.board)
        self.move_list.clear()
        self.status_label.setText("White to move")
        self.analysis_label.setText("")

    def _on_human_move(self, move):
        san = self.state.board.san(move)
        self.state.push(move)
        self.board = self.state.board
        self.move_history.append(san)
        self.redo_stack.clear()
        self.board_widget.set_board(self.board)
        self.move_list.addItem(san)
        self.status_label.setText(f"{'White' if self.board.turn == chess.WHITE else 'Black'} to move")
        # Only call AI if it's now the AI's turn and the game is not over
        ai_plays_white = False  # Set to True if you want AI to play White
        ai_plays_black = True   # Set to True if you want AI to play Black
        if not self.board.is_game_over():
            if (self.board.turn == chess.WHITE and ai_plays_white) or (self.board.turn == chess.BLACK and ai_plays_black):
                self.status_label.setText("AI thinking...")
                self.start_ai_thread()

    def start_ai_thread(self):
        from PyQt6.QtCore import QObject, QThread, pyqtSignal
        import logging
        class AiWorker(QObject):
            move_found = pyqtSignal(object, str)
            error = pyqtSignal(str)
            def __init__(self, board, move_stack, depth):
                super().__init__()
                self.board = board.copy()
                self.move_stack = list(move_stack)
                self.depth = depth
            def run(self):
                try:
                    from strong_chess_ai.core.board import GameState
                    from strong_chess_ai.core.search import find_best_move
                    state = GameState()
                    state.board = self.board
                    state.move_history = self.move_stack
                    result = find_best_move(state, max_depth=self.depth, time_limit_s=3.0)
                    if result.pv:
                        move = result.pv[0]
                        san = self.board.san(move)
                        self.move_found.emit(move, san)
                    else:
                        self.error.emit("AI could not find a move.")
                except Exception as e:
                    logging.exception(f"[DEBUG] Exception in AiWorker: {e}")
                    self.error.emit(str(e))

        self.ai_thread = QThread()
        self.ai_worker = AiWorker(self.board, self.board.move_stack, self.depth_slider.value())
        self.ai_worker.moveToThread(self.ai_thread)
        self.ai_thread.started.connect(self.ai_worker.run)
        self.ai_worker.move_found.connect(self.on_ai_move_found)
        self.ai_worker.error.connect(self.on_ai_move_error)
        self.ai_worker.move_found.connect(lambda *_: self.ai_thread.quit())
        self.ai_worker.error.connect(lambda *_: self.ai_thread.quit())
        self.ai_thread.start()

    def on_ai_move_found(self, move, san):
        self.board.push(move)
        self.move_history.append(san)
        self.board_widget.set_board(self.board)
        self.board_widget.repaint()
        self.move_list.addItem(san)
        if self.board.is_game_over():
            if self.board.is_checkmate():
                winner = "White" if self.board.turn == chess.BLACK else "Black"
                self.status_label.setText(f"Checkmate! {winner} wins.")
            elif self.board.is_stalemate():
                self.status_label.setText("Stalemate! Draw.")
            else:
                self.status_label.setText("Game over.")
        else:
            self.status_label.setText(f"{'White' if self.board.turn == chess.WHITE else 'Black'} to move")

    def on_ai_move_error(self, msg):
        self.status_label.setText(msg)
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "AI Error", msg)


    def _ai_move(self):
        import logging
        print("[DEBUG] _ai_move thread started")
        logging.warning("[DEBUG] _ai_move thread started")
        try:
            depth = self.depth_slider.value()
            from strong_chess_ai.core.board import GameState
            state = GameState()
            state.board = self.board.copy()  # Use a copy to avoid mutating GUI board during search
            state.move_history = list(self.board.move_stack)

            logging.info(f"[DEBUG] Calling find_best_move. FEN: {state.board.fen()}")
            result = find_best_move(state, max_depth=depth, time_limit_s=3.0)

            if result.pv:
                move = result.pv[0]
                san = self.board.san(move)
                logging.info(f"[DEBUG] AI move: {move.uci()} | SAN: {san}")

                def update_after_ai():
                    self.board.push(move)
                    self.move_history.append(san)
                    self.board_widget.set_board(self.board)
                    self.board_widget.repaint()
                    self.move_list.addItem(san)

                    if self.board.is_game_over():
                        if self.board.is_checkmate():
                            winner = "White" if self.board.turn == chess.BLACK else "Black"
                            self.status_label.setText(f"Checkmate! {winner} wins.")
                        elif self.board.is_stalemate():
                            self.status_label.setText("Stalemate! Draw.")
                        else:
                            self.status_label.setText("Game over.")
                    else:
                        self.status_label.setText(f"{'White' if self.board.turn == chess.WHITE else 'Black'} to move")

                QTimer.singleShot(0, update_after_ai)

            else:
                logging.warning("[DEBUG] AI could not find a move.")
                def update_no_move():
                    self.status_label.setText("AI could not find a move.")
                QTimer.singleShot(0, update_no_move)

        except Exception as e:
            logging.exception(f"[DEBUG] Exception in _ai_move: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QTimer.singleShot(0, lambda: QMessageBox.warning(self, "AI Error", str(e)))


    def _on_flip(self):
        self.board_widget.set_flipped(not self.board_widget.flipped)

    def _on_undo(self):
        if len(self.board.move_stack) > 0:
            move = self.board.pop()
            if self.move_history:
                self.redo_stack.append(self.move_history.pop())
            self.board_widget.set_board(self.board)
            self.move_list.takeItem(self.move_list.count()-1)
            self.status_label.setText(f"{'White' if self.board.turn == chess.WHITE else 'Black'} to move")

    def _on_redo(self):
        # Redo logic for popped moves
        pass

    def _on_new_game(self):
        self._reset_game()

    def _on_depth_change(self, value):
        self.depth_label.setText(f"AI Depth: {value}")
        # TODO: Update AI search depth

    def _on_dark_mode(self):
        # Toggle dark mode
        pass

    def _on_export_pgn(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Export PGN", os.getcwd(), "PGN Files (*.pgn)")
        if fname:
            game = chess.pgn.Game()
            node = game
            for move in self.board.move_stack:
                node = node.add_variation(move)
            with open(fname, "w") as f:
                print(game, file=f)

    def _on_import_pgn(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Import PGN", os.getcwd(), "PGN Files (*.pgn)")
        if fname:
            with open(fname, "r") as f:
                game = chess.pgn.read_game(f)
            self._reset_game()
            board = chess.Board()
            for move in game.mainline_moves():
                board.push(move)
                san = board.san(move)
                self.move_list.addItem(san)
                self.move_history.append(san)
            self.board = board
            self.board_widget.set_board(self.board)
            self.status_label.setText(f"{'White' if self.board.turn == chess.WHITE else 'Black'} to move")

    def _on_hint(self):
        # TODO: Integrate with AI for best move
        QMessageBox.information(self, "Hint", "Hint feature coming soon!")

    def _on_move_selected(self, item):
        # TODO: Allow navigation to selected move
        pass

# --- Entry Point ---
def main():
    app = QApplication(sys.argv)
    win = ChessMainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
