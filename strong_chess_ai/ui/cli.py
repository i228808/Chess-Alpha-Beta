import chess
import logging
from strong_chess_ai.core.search import find_best_move
from strong_chess_ai.core.board import GameState

UNICODE_PIECES = {
    'P': '\u2659', 'N': '\u2658', 'B': '\u2657', 'R': '\u2656', 'Q': '\u2655', 'K': '\u2654',
    'p': '\u265F', 'n': '\u265E', 'b': '\u265D', 'r': '\u265C', 'q': '\u265B', 'k': '\u265A',
    None: ' '
}

def print_board(board):
    print('  a b c d e f g h')
    for rank in range(7, -1, -1):
        row = [str(rank+1)]
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            row.append(UNICODE_PIECES[piece.symbol()] if piece else '.')
        print(' '.join(row))
    print('  a b c d e f g h')
    print('FEN:', board.fen())
    print()

def main():
    logging.basicConfig(level=logging.INFO)
    print('Welcome to Strong Chess AI CLI!')
    color = input('Play as (w/b): ').strip().lower()
    human_white = color == 'w'
    state = GameState()
    move_history = []
    while not state.is_terminal():
        print_board(state.board)
        if state.board.turn == chess.WHITE and human_white or state.board.turn == chess.BLACK and not human_white:
            move_str = input('Your move (UCI or SAN): ').strip()
            move_str = move_str.strip()
            if move_str.lower() in ("help", "?", "moves"):
                print("Legal moves:")
                for move in state.board.legal_moves:
                    try:
                        san = state.board.san(move)
                    except Exception:
                        san = move.uci()
                    print(f"  {san} ({move.uci()})")
                continue
            if len(move_str) == 4 or (len(move_str) == 5 and move_str[-1] in 'qrbnQRBN'):
                # Try UCI first
                try:
                    move = chess.Move.from_uci(move_str.lower())
                    if move not in state.board.legal_moves:
                        print('Illegal move. Try again. Type "help" to see legal moves.')
                        continue
                    san = state.board.san(move)
                    logging.info(f"[MOVE LOG] Human: {san} ({move.uci()}) | FEN before: {state.board.fen()}")
                    state.push(move)
                    logging.info(f"[MOVE LOG] Human: {san} ({move.uci()}) | FEN after: {state.board.fen()}")
                    move_history.append(san)
                    continue
                except Exception as e:
                    pass  # Try SAN next
            # Try SAN
            try:
                move = state.board.parse_san(move_str)
                if move not in state.board.legal_moves:
                    print('Illegal move. Try again. Type "help" to see legal moves.')
                    continue
                san = state.board.san(move)
                logging.info(f"[MOVE LOG] Human: {san} ({move.uci()}) | FEN before: {state.board.fen()}")
                state.push(move)
                logging.info(f"[MOVE LOG] Human: {san} ({move.uci()}) | FEN after: {state.board.fen()}")
                move_history.append(san)
            except Exception as e:
                print('Invalid move. Type "help" to see legal moves.')
                continue
        else:
            print('AI thinking...')
            result = find_best_move(state, max_depth=4, time_limit_s=3.0, threads=1)
            if result and result.pv:
                move = result.pv[0]
                if move not in state.board.legal_moves:
                    print('AI produced illegal move:', move.uci())
                    break
                san = state.board.san(move)  # Always get SAN before push
                logging.info(f"[MOVE LOG] AI: {san} ({move.uci()}) | FEN before: {state.board.fen()}")
                state.push(move)
                logging.info(f"[MOVE LOG] AI: {san} ({move.uci()}) | FEN after: {state.board.fen()}")
                move_history.append(san)
            else:
                print('AI could not find a move.')
                break
    print_board(state.board)
    print('Game over! Result:', state.result())
    print('Move history:', ' '.join(move_history))

if __name__ == '__main__':
    main()
