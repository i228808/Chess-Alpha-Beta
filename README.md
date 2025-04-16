# Assignment 3: Solve Chess via Alpha-Beta Pruning

**Name:** Abdullah Mansoor  
**Roll Number:** i228808  
**Course:** AAI  

---

## Overview
This project is a chess engine and GUI that demonstrates solving chess using iterative deepening and alpha-beta pruning. You can play against the AI using either a PyQt6-based graphical interface or a command-line interface.

---

## Requirements
- Python 3.10+
- Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. **Graphical User Interface (GUI)**

Launch the PyQt6 GUI:

```bash
python -m strong_chess_ai.ui.qt_gui
```

**Features:**
- Play against the AI with adjustable search depth and time limit (use the sliders on the right panel).
- Undo/Redo, flip board, import/export PGN, and get move hints.
- The window title and headers include your name, roll number, and assignment info.

### 2. **Command-Line Interface (CLI)**

Run the CLI chess game:

```bash
python -m strong_chess_ai.ui.cli
```

**Features:**
- Play against the AI in your terminal.
- Enter moves in UCI or SAN format (e.g., e2e4 or Nf3).
- Type `help` or `moves` to see all legal moves.

---

## Project Structure

```
strong_chess_ai/
├── core/
│   ├── bitboard.py
│   ├── board.py
│   ├── book.py
│   ├── eval.py
│   ├── search.py
│   ├── tt.py
│   └── zobrist.py
├── ui/
│   ├── qt_gui.py      # Main GUI
│   ├── cli.py         # Command-line interface
│   ├── controller.py  # Game controller
│   └── resources.py   # UI constants/assets
└── requirements.txt
```

---

## Notes
- Make sure to keep all files in their respective folders as shown above.
- If you encounter any issues, ensure all dependencies are installed and you are using a compatible Python version.
- For assignment submission, upload only the essential files as described.

---

**Good luck and happy chess solving!**
