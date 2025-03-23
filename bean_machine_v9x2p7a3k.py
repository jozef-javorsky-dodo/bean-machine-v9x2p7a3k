from __future__ import annotations
import logging
from collections.abc import Iterator, MutableSequence, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from random import gauss, random
from sys import argv
from typing import Final, Tuple, TypeAlias
from PIL import Image, ImageDraw
import unittest
import os
from uuid import uuid4

Position: TypeAlias = float
Frequency: TypeAlias = int
Color: TypeAlias = Tuple[int, int, int]

@dataclass(frozen=True)
class BoardConfig:
    NUM_ROWS: Final[int] = 12
    NUM_BALLS: Final[int] = 100000
    PROGRESS_DIVISIONS: Final[int] = 20
    BOARD_WIDTH: Final[int] = 700
    BOARD_HEIGHT: Final[int] = 500
    PEG_RADIUS: Final[int] = 4
    DAMPING_FACTOR: Final[float] = 0.8
    ELASTICITY: Final[float] = 0.7
    INITIAL_VARIANCE: Final[float] = 2.0
    MIN_BOUNCE_PROBABILITY: Final[float] = 0.2
    MAX_BOUNCE_PROBABILITY: Final[float] = 0.8
    BOUNCE_DISTANCE_FACTOR: Final[float] = 0.1
    HALF_PEG_RADIUS_FACTOR: Final[float] = 0.5
    BOUNCE_PROB_CENTER: Final[float] = 0.5
    BACKGROUND_COLOR: Final[Color] = (102, 51, 153)
    LEFT_COLOR: Final[Color] = (122, 122, 244)
    RIGHT_COLOR: Final[Color] = (122, 244, 122)
    SMOOTHING_WINDOW: Final[int] = 3
    HISTOGRAM_BAR_MIN_WIDTH: Final[int] = 1
    BOUNCE_PROB_CACHE_SIZE: Final[int] = 128
    DEFAULT_IMAGE_FILENAME: Final[str] = f"galton_board_{uuid4().hex[:8]}.png"
    LOG_FORMAT: Final[str] = "%(levelname)s: %(message)s"

@dataclass
class GaltonBoard:
    num_rows: int = field(default=BoardConfig.NUM_ROWS)
    num_balls: int = field(default=BoardConfig.NUM_BALLS)
    board_width: int = field(default=BoardConfig.BOARD_WIDTH)
    board_height: int = field(default=BoardConfig.BOARD_HEIGHT)
    slot_counts: MutableSequence[Frequency] = field(default_factory=list)
    image: Image.Image = field(init=False)
    draw: ImageDraw.Draw = field(init=False)
    _lower_bound: Position = field(init=False)
    _upper_bound: Position = field(init=False)
    _board_center: Position = field(init=False)
    _half_board_width: int = field(init=False)

    def __post_init__(self) -> None:
        self.slot_counts = [0] * self.board_width
        self.image = Image.new("RGB", (self.board_width, self.board_height),
                                BoardConfig.BACKGROUND_COLOR)
        self.draw = ImageDraw.Draw(self.image)
        self._lower_bound = BoardConfig.PEG_RADIUS
        self._upper_bound = self.board_width - BoardConfig.PEG_RADIUS
        self._board_center = self.board_width / 2
        self._half_board_width = self.board_width // 2
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        dims = (self.num_rows, self.num_balls, self.board_width, self.board_height)
        if not all(x > 0 for x in dims):
            raise ValueError("Dimensions must be positive.")
        if self.board_width <= 2 * BoardConfig.PEG_RADIUS:
            raise ValueError("Board width must be greater than twice peg radius.")

    def simulate(self) -> None:
        slots = [0] * self.board_width
        progress_step = max(1, self.num_balls // BoardConfig.PROGRESS_DIVISIONS)
        for idx, slot in enumerate(self._generate_ball_paths(), start=1):
            slots[slot] += 1
            if idx % progress_step == 0:
                logging.info(f"Simulated {idx}/{self.num_balls} balls.")
        self._apply_smoothing(slots)

    def _generate_ball_paths(self) -> Iterator[int]:
        diam = BoardConfig.PEG_RADIUS * 2
        half_offset = BoardConfig.PEG_RADIUS * BoardConfig.HALF_PEG_RADIUS_FACTOR
        min_prob = BoardConfig.MIN_BOUNCE_PROBABILITY
        max_prob = BoardConfig.MAX_BOUNCE_PROBABILITY
        damp = BoardConfig.DAMPING_FACTOR
        elast = BoardConfig.ELASTICITY
        bounce_ctr = BoardConfig.BOUNCE_PROB_CENTER
        bounce_fact = BoardConfig.BOUNCE_DISTANCE_FACTOR
        var = BoardConfig.INITIAL_VARIANCE
        inv_rad = 1.0 / BoardConfig.PEG_RADIUS
        for _ in range(self.num_balls):
            pos = self._board_center + gauss(0, var)
            mom = 0.0
            toggle = 0
            for _ in range(self.num_rows):
                off = toggle * half_offset
                toggle = 1 - toggle
                peg_pos = pos + off
                delta = (pos - peg_pos) * inv_rad
                prob = self._calculate_bounce_probability(
                    delta, bounce_ctr, bounce_fact, min_prob, max_prob)
                direction = 1 if random() < prob else -1
                force = (1.0 - abs(delta)) * elast
                mom = mom * damp + direction * force * diam
                pos = self._constrain_position(pos + mom)
            yield int(pos)

    @staticmethod
    @lru_cache(maxsize=BoardConfig.BOUNCE_PROB_CACHE_SIZE)
    def _calculate_bounce_probability(
        delta: float, center: float, factor: float,
        min_prob: float, max_prob: float
    ) -> float:
        p = center + factor * delta
        return max(min_prob, min(max_prob, p))

    def _constrain_position(self, pos: Position) -> Position:
        return max(self._lower_bound, min(self._upper_bound, pos))

    def _apply_smoothing(self, slots: Sequence[Frequency]) -> None:
        window = BoardConfig.SMOOTHING_WINDOW
        total = len(slots)
        for i in range(len(self.slot_counts)):
            start = max(0, i - window)
            end = min(total, i + window + 1)
            seg = slots[start:end]
            if (length := end - start) > 0:
                self.slot_counts[i] = sum(seg) // length

    def generate_image(self) -> Image.Image:
        m = max(self.slot_counts, default=0)
        if m:
            bar_width = max(BoardConfig.HISTOGRAM_BAR_MIN_WIDTH,
                            self.board_width // len(self.slot_counts))
            self._draw_all_bars(m, bar_width)
        return self.image

    def _draw_all_bars(self, m: int, bar_width: int) -> None:
        left = BoardConfig.LEFT_COLOR
        right = BoardConfig.RIGHT_COLOR
        for i, freq in enumerate(self.slot_counts):
            x = i * bar_width
            col = left if x < self._half_board_width else right
            self._draw_histogram_bar(i, freq, m, bar_width, col)

    def _draw_histogram_bar(
        self, i: int, freq: int, m: int, bar_width: int, col: Color
    ) -> None:
        if not m:
            return
        bar_height = int(freq / m * self.board_height)
        xs = i * bar_width
        xe = xs + bar_width
        ys = self.board_height - bar_height
        self.draw.rectangle([xs, ys, xe, self.board_height], fill=col)

    def save_image(self, filename: str = BoardConfig.DEFAULT_IMAGE_FILENAME) -> None:
        try:
            path = Path(filename).resolve()
            self.generate_image().save(path)
        except (OSError, IOError) as err:
            logging.error(f"Failed to save image to {filename}: {err}.")
            raise

def generate_galton_board() -> None:
    gb = GaltonBoard()
    gb.simulate()
    gb.save_image()

def main() -> None:
    logging.basicConfig(level=logging.INFO, format=BoardConfig.LOG_FORMAT)
    try:
        generate_galton_board()
    except Exception as exc:
        logging.exception(f"Fatal error: {exc}.")
        raise

class TestGaltonBoard(unittest.TestCase):
    def test_simulation(self):
        gb = GaltonBoard(num_rows=5, num_balls=500, board_width=300, board_height=200)
        gb.simulate()
        self.assertEqual(len(gb.slot_counts), gb.board_width)
        self.assertTrue(all(isinstance(c, int) for c in gb.slot_counts))

    def test_generate_image(self):
        gb = GaltonBoard(num_rows=5, num_balls=500, board_width=300, board_height=200)
        gb.simulate()
        img = gb.generate_image()
        self.assertEqual(img.size, (gb.board_width, gb.board_height))

    def test_save_image(self):
        fname = "test_output.png"
        gb = GaltonBoard(num_rows=5, num_balls=500, board_width=300, board_height=200)
        gb.simulate()
        gb.save_image(fname)
        self.assertTrue(os.path.exists(fname))
        os.remove(fname)

if __name__ == "__main__":
    if "--test" in argv:
        unittest.main(argv=[argv[0]])
    else:
        main()