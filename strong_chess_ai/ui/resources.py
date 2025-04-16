import os
from PIL import Image, ImageTk
import tkinter as tk
import wave
import contextlib

# Directory containing asset files (images and sounds)
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Board and UI dimensions
SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
SIDE_PANEL_WIDTH = 200

# Board colors for light and dark squares
LIGHT = "#f0d9b5"
DARK = "#b58863"

# Unicode chess piece symbols (fallback if images are not used)
UNICODE_PIECES = {
    'P': '\u2659', 'N': '\u2658', 'B': '\u2657', 'R': '\u2656', 'Q': '\u2655', 'K': '\u2654',
    'p': '\u265F', 'n': '\u265E', 'b': '\u265D', 'r': '\u265C', 'q': '\u265B', 'k': '\u265A',
}

# Predefined piece types for image loading.
PIECE_NAMES = ["P", "N", "B", "R", "Q", "K"]
PIECE_IMAGES = {}

# Determine the best available resampling filter: use LANCZOS if available.
try:
    resample_filter = Image.Resampling.LANCZOS
except AttributeError:
    resample_filter = Image.ANTIALIAS

# Load piece images from ASSETS_DIR if available.
for color in ["w", "b"]:
    for name in PIECE_NAMES:
        fname = f"{color}{name}.png"
        path = os.path.join(ASSETS_DIR, fname)
        if os.path.exists(path):
            try:
                img = Image.open(path).resize((SQUARE_SIZE, SQUARE_SIZE), resample_filter)
                PIECE_IMAGES[color + name] = ImageTk.PhotoImage(img)
            except Exception as e:
                print(f"Warning: Failed to load image '{path}': {e}")

# Sound resource file paths.
SOUND_MOVE = os.path.join(ASSETS_DIR, "move.wav")
SOUND_CAPTURE = os.path.join(ASSETS_DIR, "capture.wav")

def play_sound(path: str) -> None:
    """
    Play a sound from the given file path using the simpleaudio library.
    If sound playback fails, the function silently ignores the error.
    
    Args:
        path (str): The file path to the sound file.
    """
    try:
        import simpleaudio as sa
        wave_obj = sa.WaveObject.from_wave_file(path)
        wave_obj.play()
    except Exception:
        # Sound playback failed; silently ignore.
        pass
