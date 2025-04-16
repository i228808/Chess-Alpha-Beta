import os
from PIL import Image, ImageTk
import tkinter as tk
import wave
import contextlib

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
SIDE_PANEL_WIDTH = 200
LIGHT = "#f0d9b5"
DARK = "#b58863"
UNICODE_PIECES = {
    'P': '\u2659', 'N': '\u2658', 'B': '\u2657', 'R': '\u2656', 'Q': '\u2655', 'K': '\u2654',
    'p': '\u265F', 'n': '\u265E', 'b': '\u265D', 'r': '\u265C', 'q': '\u265B', 'k': '\u265A',
}

PIECE_NAMES = ["P", "N", "B", "R", "Q", "K"]
PIECE_IMAGES = {}
for color in ["w", "b"]:
    for name in PIECE_NAMES:
        fname = f"{color}{name}.png"
        path = os.path.join(ASSETS_DIR, fname)
        if os.path.exists(path):
            img = Image.open(path).resize((SQUARE_SIZE, SQUARE_SIZE), Image.ANTIALIAS)
            PIECE_IMAGES[color + name] = ImageTk.PhotoImage(img)

# Sound resources
SOUND_MOVE = os.path.join(ASSETS_DIR, "move.wav")
SOUND_CAPTURE = os.path.join(ASSETS_DIR, "capture.wav")

def play_sound(path):
    try:
        import simpleaudio as sa
        wave_obj = sa.WaveObject.from_wave_file(path)
        wave_obj.play()
    except Exception:
        pass  # Fallback: do nothing (no pop-up window!)
