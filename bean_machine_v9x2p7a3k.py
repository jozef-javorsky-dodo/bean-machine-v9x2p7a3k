from __future__ import annotations

import logging
from collections.abc import Iterator, MutableSequence, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from random import gauss, random
from typing import Final, Tuple, TypeAlias

from PIL import Image, ImageDraw

Position: TypeAlias = float
Frequency: TypeAlias = int
Color: TypeAlias = Tuple[int, int, int]


@dataclass(frozen=True)
class BoardConfig:
    NUM_ROWS: Final[int] = 12
    NUM_BALLS: Final[int] = 100_000
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
    DEFAULT_IMAGE_FILENAME: Final[str] = "galton_board.png"
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
        self.image = Image.new(
            "RGB",
            (self.board_width, self.board_height),
            BoardConfig.BACKGROUND_COLOR
        )
        self.draw = ImageDraw.Draw(self.image)
        self._lower_bound = BoardConfig.PEG_RADIUS
        self._upper_bound = self.board_width - BoardConfig.PEG_RADIUS
        self._board_center = self.board_width / 2
        self._half_board_width = self.board_width // 2
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        dimensions = (self.num_rows, self.num_balls,
                      self.board_width, self.board_height)
        if not all(dim > 0 for dim in dimensions):
            raise ValueError("Dimensions must be positive.")
        if self.board_width <= 2 * BoardConfig.PEG_RADIUS:
            raise ValueError("Board width must be greater than twice peg radius.")

    def simulate(self) -> None:
        slots = [0] * self.board_width
        progress_step = max(1, self.num_balls // BoardConfig.PROGRESS_DIVISIONS)
        
        for idx, slot_index in enumerate(self._generate_ball_paths(), start=1):
            slots[slot_index] += 1
            if idx % progress_step == 0:
                logging.info(f"Simulated {idx}/{self.num_balls} balls.")
                
        self._apply_smoothing(slots)

    def _generate_ball_paths(self) -> Iterator[int]:
        peg_diameter = BoardConfig.PEG_RADIUS * 2
        half_peg_offset = BoardConfig.PEG_RADIUS * BoardConfig.HALF_PEG_RADIUS_FACTOR
        min_bounce_prob = BoardConfig.MIN_BOUNCE_PROBABILITY
        max_bounce_prob = BoardConfig.MAX_BOUNCE_PROBABILITY
        damping = BoardConfig.DAMPING_FACTOR
        elasticity = BoardConfig.ELASTICITY
        bounce_center = BoardConfig.BOUNCE_PROB_CENTER
        bounce_factor = BoardConfig.BOUNCE_DISTANCE_FACTOR
        initial_variance = BoardConfig.INITIAL_VARIANCE
        inv_peg_radius = 1.0 / BoardConfig.PEG_RADIUS
        
        for _ in range(self.num_balls):
            position = self._board_center + gauss(0, initial_variance)
            momentum = 0.0
            
            for row_idx in range(self.num_rows):
                offset = (row_idx % 2) * half_peg_offset
                peg_position = position + offset
                delta = (position - peg_position) * inv_peg_radius
                
                bounce_probability = self._calculate_bounce_probability(
                    delta, bounce_center, bounce_factor,
                    min_bounce_prob, max_bounce_prob
                )
                direction = 1 if random() < bounce_probability else -1
                force = (1.0 - abs(delta)) * elasticity
                
                momentum = momentum * damping + direction * force * peg_diameter
                position = self._constrain_position(position + momentum)
                
            yield int(position)

    @staticmethod
    @lru_cache(maxsize=BoardConfig.BOUNCE_PROB_CACHE_SIZE)
    def _calculate_bounce_probability(
        delta: float, center: float, factor: float,
        min_prob: float, max_prob: float
    ) -> float:
        probability = center + factor * delta
        return max(min_prob, min(max_prob, probability))

    def _constrain_position(self, position: Position) -> Position:
        return max(self._lower_bound, min(self._upper_bound, position))

    def _apply_smoothing(self, slots: Sequence[Frequency]) -> None:
        window = BoardConfig.SMOOTHING_WINDOW
        slot_length = len(slots)
        
        for idx in range(len(self.slot_counts)):
            start_idx = max(0, idx - window)
            end_idx = min(slot_length, idx + window + 1)
            
            if (segment_length := end_idx - start_idx) > 0:
                segment = slots[start_idx:end_idx]
                self.slot_counts[idx] = sum(segment) // segment_length

    def generate_image(self) -> Image.Image:
        max_frequency = max(self.slot_counts, default=0)
        
        if max_frequency:
            bar_width = max(
                BoardConfig.HISTOGRAM_BAR_MIN_WIDTH,
                self.board_width // len(self.slot_counts)
            )
            self._draw_all_bars(max_frequency, bar_width)
            
        return self.image

    def _draw_all_bars(self, max_frequency: int, bar_width: int) -> None:
        left_color = BoardConfig.LEFT_COLOR
        right_color = BoardConfig.RIGHT_COLOR
        
        for idx, frequency in enumerate(self.slot_counts):
            x_position = idx * bar_width
            color = left_color if x_position < self._half_board_width else right_color
            
            self._draw_histogram_bar(
                idx, frequency, max_frequency, bar_width, color
            )

    def _draw_histogram_bar(
        self, idx: int, frequency: int, max_frequency: int,
        bar_width: int, color: Color
    ) -> None:
        if not max_frequency:
            return
            
        bar_height = int(frequency / max_frequency * self.board_height)
        x_start = idx * bar_width
        x_end = x_start + bar_width
        y_start = self.board_height - bar_height
        
        self.draw.rectangle(
            [x_start, y_start, x_end, self.board_height],
            fill=color
        )

    def save_image(self, filename: str = BoardConfig.DEFAULT_IMAGE_FILENAME) -> None:
        try:
            output_path = Path(filename).resolve()
            self.generate_image().save(output_path)
        except (IOError, OSError) as error:
            logging.error(f"Failed to save image to {filename}: {error}.")
            raise


def generate_galton_board() -> None:
    board = GaltonBoard()
    board.simulate()
    board.save_image()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=BoardConfig.LOG_FORMAT)
    try:
        generate_galton_board()
    except Exception as error:
        logging.exception(f"Fatal error: {error}.")
        raise


if __name__ == "__main__":
    main()