"""
Screen Chess Adapter for Universal Jev-GamePilot.
Calibrates on an 8x8 chessboard on screen (e.g. Chess.com or Lichess),
tracks board state, detects opponent moves via square highlights,
queries Jev System One for tactical move decisions, and executes clicks.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import chess
import cv2
import numpy as np
from typesafe_sdk import Choice, Score, TypeSafeClient


@dataclass
class ChessBoardCalibration:
    left: int
    top: int
    width: int
    height: int
    orientation: str = "white"  # "white" (a1 bottom-left) or "black" (h8 bottom-left)


class ScreenChessAdapter:
    def __init__(self, calibration: Optional[ChessBoardCalibration] = None):
        self.calib = calibration or ChessBoardCalibration(
            left=300, top=150, width=560, height=560, orientation="white"
        )
        self.board = chess.Board()
        self.last_frame_bgr: Optional[np.ndarray] = None
        self.last_move_squares: List[str] = []
        self.is_my_turn = True

    def reset_board(self, fen: str = chess.STARTING_FEN):
        self.board = chess.Board(fen)
        self.last_move_squares.clear()

    def set_calibration(
        self, left: int, top: int, width: int, height: int, orientation="white"
    ):
        self.calib = ChessBoardCalibration(
            left=left,
            top=top,
            width=width,
            height=height,
            orientation=orientation,
        )

    def get_square_bounds(self, square: chess.Square) -> Tuple[int, int, int, int]:
        """Returns (x, y, w, h) relative to the captured chessboard image."""
        file_idx = chess.square_file(square)  # 0 to 7 (a to h)
        rank_idx = chess.square_rank(square)  # 0 to 7 (1 to 8)

        sq_w = self.calib.width / 8.0
        sq_h = self.calib.height / 8.0

        if self.calib.orientation == "white":
            col = file_idx
            row = 7 - rank_idx
        else:
            col = 7 - file_idx
            row = rank_idx

        x = int(col * sq_w)
        y = int(row * sq_h)
        return (x, y, int(sq_w), int(sq_h))

    def get_square_screen_center(self, square: chess.Square) -> Tuple[int, int]:
        """Returns absolute screen (X, Y) coordinates for clicking."""
        rel_x, rel_y, sq_w, sq_h = self.get_square_bounds(square)
        abs_x = int(self.calib.left + rel_x + sq_w / 2.0)
        abs_y = int(self.calib.top + rel_y + sq_h / 2.0)
        return (abs_x, abs_y)

    def detect_highlighted_squares(
        self, board_frame_bgr: np.ndarray
    ) -> List[chess.Square]:
        """
        Detects the move highlight squares on Chess.com / Lichess.
        Chess.com highlights last move squares with yellow/green tints.
        """
        hsv = cv2.cvtColor(board_frame_bgr, cv2.COLOR_BGR2HSV)
        # Yellow / lime highlight mask (hue ~20 to 55)
        lower_yellow = np.array([20, 40, 150])
        upper_yellow = np.array([55, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        sq_scores = []
        for sq in chess.SQUARES:
            x, y, w, h = self.get_square_bounds(sq)
            sq_roi = mask[y : y + h, x : x + w]
            density = float(np.mean(sq_roi))
            if density > 15.0:
                sq_scores.append((density, sq))

        sq_scores.sort(key=lambda item: item[0], reverse=True)
        return [sq for _, sq in sq_scores[:2]]

    def get_top_candidate_moves(
        self, max_candidates: int = 6
    ) -> List[chess.Move]:
        """Filters and orders legal candidate moves for Jev to evaluate."""
        legal = list(self.board.legal_moves)
        if not legal:
            return []

        # Heuristic sort: captures and checks first, then center development
        def move_weight(m: chess.Move) -> int:
            w = 0
            if self.board.is_capture(m):
                w += 10
            if self.board.gives_check(m):
                w += 8
            to_sq = m.to_square
            # Center squares e4, d4, e5, d5 bonus
            if to_sq in [chess.E4, chess.D4, chess.E5, chess.D5]:
                w += 5
            return w

        legal.sort(key=move_weight, reverse=True)
        return legal[:max_candidates]

    def query_jev_best_move(
        self, client: TypeSafeClient, model: str = "jev-latest"
    ) -> Optional[Dict[str, Any]]:
        """
        Queries TypeSafe Jev System One to evaluate the chess position and choose the best move.
        """
        candidates = self.get_top_candidate_moves(max_candidates=6)
        if not candidates:
            return None

        candidate_sans = [self.board.san(m) for m in candidates]
        move_map = {self.board.san(m): m for m in candidates}

        state_dict = {
            "game": "Chess",
            "fen": self.board.fen(),
            "turn": "White" if self.board.turn == chess.WHITE else "Black",
            "fullmove_number": self.board.fullmove_number,
            "is_check": self.board.is_check(),
            "legal_candidate_moves": candidate_sans,
        }

        criteria = {}
        for san in candidate_sans:
            m = move_map[san]
            desc = f"Play {san}"
            if self.board.is_capture(m):
                desc += " (capture piece)"
            if self.board.gives_check(m):
                desc += " (gives check to enemy King)"
            criteria[san] = desc

        questions = {
            "best_tactical_move": Choice(
                instructions=f"Select the most accurate, principled chess move for {'White' if self.board.turn == chess.WHITE else 'Black'}.",
                criteria=criteria,
            ),
            "position_evaluation": Score(
                instructions="Evaluate the position advantage from 0 (losing) to 4 (winning)",
                criteria=[
                    "0: Losing / severe disadvantage",
                    "1: Slight disadvantage",
                    "2: Balanced / equal game",
                    "3: Clear positional advantage",
                    "4: Decisive winning attack or large material advantage",
                ],
            ),
        }

        try:
            t0 = time.perf_counter()
            resp = client.system_one(
                state=state_dict, model=model, questions=questions
            )
            dt = (time.perf_counter() - t0) * 1000.0

            ans = resp.answers
            chosen_san = getattr(
                ans["best_tactical_move"], "choice", candidate_sans[0]
            )
            chosen_conf = getattr(
                ans["best_tactical_move"], "confidence", 0.95
            )
            eval_score = float(
                getattr(ans["position_evaluation"], "score", 2.0)
            )

            chosen_move = move_map.get(chosen_san, candidates[0])

            return {
                "san": chosen_san,
                "uci": chosen_move.uci(),
                "move": chosen_move,
                "from_square": chosen_move.from_square,
                "to_square": chosen_move.to_square,
                "from_coords": self.get_square_screen_center(
                    chosen_move.from_square
                ),
                "to_coords": self.get_square_screen_center(
                    chosen_move.to_square
                ),
                "eval": round(eval_score, 2),
                "confidence": round(chosen_conf, 3),
                "latency_ms": round(dt, 1),
            }

        except Exception as e:
            print(f"[ScreenChessAdapter] Jev query error: {e}")
            return None

    def render_board_overlay(
        self, board_frame_bgr: np.ndarray, recommended_move: Optional[Dict] = None
    ) -> np.ndarray:
        """Renders 8x8 grid lines, coordinates, and best move arrow on the frame."""
        annotated = board_frame_bgr.copy()
        sq_w = self.calib.width / 8.0
        sq_h = self.calib.height / 8.0

        # Draw 8x8 grid lines
        for i in range(9):
            x = int(i * sq_w)
            y = int(i * sq_h)
            cv2.line(
                annotated,
                (x, 0),
                (x, int(self.calib.height)),
                (0, 255, 204),
                1,
            )
            cv2.line(
                annotated, (0, y), (int(self.calib.width), y), (0, 255, 204), 1
            )

        # Draw recommended move arrow
        if recommended_move:
            from_sq = recommended_move["from_square"]
            to_sq = recommended_move["to_square"]
            fx, fy, _, _ = self.get_square_bounds(from_sq)
            tx, ty, _, _ = self.get_square_bounds(to_sq)
            pt1 = (int(fx + sq_w / 2), int(fy + sq_h / 2))
            pt2 = (int(tx + sq_w / 2), int(ty + sq_h / 2))

            # Arrow in neon green/cyan
            cv2.arrowedLine(annotated, pt1, pt2, (0, 255, 0), 3, tipLength=0.3)
            label = f"JEV: {recommended_move['san']} (Eval: {recommended_move['eval']}/4)"
            cv2.putText(
                annotated,
                label,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 204),
                2,
            )

        return annotated
