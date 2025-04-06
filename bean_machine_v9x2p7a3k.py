# fmt: off
from __future__ import annotations

import datetime
import logging
import math
from collections.abc import Iterator, MutableSequence, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from random import gauss, random
from typing import Final, Optional, Tuple, TypeAlias

from PIL import Image, ImageDraw

Position: TypeAlias = float
Frequency: TypeAlias = int
Color: TypeAlias = Tuple[int, int, int]
# fmt: on


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
    DEFAULT_IMAGE_BASENAME: Final[str] = "galton_board"
    LOG_FORMAT: Final[str] = "%(levelname)s: %(message)s"


@dataclass
class GaltonBoard:
    num_rows: int = field(default=BoardConfig.NUM_ROWS)
    num_balls: int = field(default=BoardConfig.NUM_BALLS)
    board_width: int = field(default=BoardConfig.BOARD_WIDTH)
    board_height: int = field(default=BoardConfig.BOARD_HEIGHT)
    slot_counts: MutableSequence[Frequency] = field(default_factory=list)
    image: Optional[Image.Image] = field(init=False, default=None)
    draw: Optional[ImageDraw.Draw] = field(init=False, default=None)
    _lower_bound: Position = field(init=False)
    _upper_bound: Position = field(init=False)
    _board_center: Position = field(init=False)
    _half_board_width_pixels: int = field(init=False)
    _peg_diameter: int = field(init=False)
    _half_peg_offset: float = field(init=False)
    _inv_peg_radius: float = field(init=False)
    _simulation_complete: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._validate_parameters()
        self.slot_counts = [0] * self.board_width
        self._lower_bound = float(BoardConfig.PEG_RADIUS)
        self._upper_bound = float(self.board_width - BoardConfig.PEG_RADIUS)
        self._board_center = self.board_width / 2.0
        self._half_board_width_pixels = self.board_width // 2
        self._peg_diameter = BoardConfig.PEG_RADIUS * 2
        self._half_peg_offset = (
            BoardConfig.PEG_RADIUS * BoardConfig.HALF_PEG_RADIUS_FACTOR
        )
        peg_radius = float(BoardConfig.PEG_RADIUS)
        self._inv_peg_radius = 1.0 / peg_radius if peg_radius > 0 else math.inf
        self._reset_state()

    def _reset_state(self) -> None:
        self.slot_counts = [0] * self.board_width
        self.image = None
        self.draw = None
        self._simulation_complete = False

    def _validate_parameters(self) -> None:
        if not (
            self.num_rows > 0
            and self.num_balls > 0
            and self.board_width > 0
            and self.board_height > 0
            and BoardConfig.PEG_RADIUS > 0
        ):
            raise ValueError("Dimensions, counts, and radius must be positive.")
        if self.board_width <= 2 * BoardConfig.PEG_RADIUS:
            raise ValueError("Board width too small for peg radius.")
        if not (0.0 <= BoardConfig.DAMPING_FACTOR <= 1.0):
            raise ValueError("Damping factor must be in [0.0, 1.0].")
        if not (0.0 <= BoardConfig.ELASTICITY <= 1.0):
            raise ValueError("Elasticity must be in [0.0, 1.0].")
        if BoardConfig.SMOOTHING_WINDOW < 0:
            raise ValueError("Smoothing window cannot be negative.")
        min_p, max_p = (
            BoardConfig.MIN_BOUNCE_PROBABILITY,
            BoardConfig.MAX_BOUNCE_PROBABILITY,
        )
        if not (0.0 <= min_p <= 1.0 and 0.0 <= max_p <= 1.0 and min_p <= max_p):
            raise ValueError("Invalid bounce probability range.")

    def simulate(self) -> None:
        if self._simulation_complete:
            logging.warning("Re-running simulation; previous results cleared.")
            self._reset_state()

        slots = [0] * self.board_width
        progress_step = max(1, self.num_balls // BoardConfig.PROGRESS_DIVISIONS)
        balls_processed = 0

        path_generator = self._generate_ball_paths()

        for _ in range(self.num_balls):
            try:
                slot_index = next(path_generator)
                clamped_index = max(0, min(self.board_width - 1, slot_index))
                slots[clamped_index] += 1
                balls_processed += 1
            except StopIteration:
                logging.error("Simulation ended prematurely: path generator empty.")
                break

            if balls_processed % progress_step == 0:
                logging.info(f"Simulated {balls_processed}/{self.num_balls} balls.")

        if balls_processed != self.num_balls:
             logging.warning(f"Only {balls_processed}/{self.num_balls} simulated.")
        elif balls_processed % progress_step != 0:
             logging.info(f"Simulated {balls_processed}/{self.num_balls} balls.")


        self._apply_smoothing(slots)
        self._simulation_complete = True

    def _generate_ball_paths(self) -> Iterator[int]:
        cfg = BoardConfig
        num_balls, num_rows = self.num_balls, self.num_rows
        board_center = self._board_center
        initial_variance = cfg.INITIAL_VARIANCE
        damping = cfg.DAMPING_FACTOR
        elasticity = cfg.ELASTICITY
        peg_diameter = float(self._peg_diameter)
        inv_peg_radius = self._inv_peg_radius
        half_peg_offset = self._half_peg_offset
        bounce_center = cfg.BOUNCE_PROB_CENTER
        bounce_factor = cfg.BOUNCE_DISTANCE_FACTOR
        min_prob, max_prob = cfg.MIN_BOUNCE_PROBABILITY, cfg.MAX_BOUNCE_PROBABILITY
        lower_bound, upper_bound = self._lower_bound, self._upper_bound

        for _ in range(num_balls):
            position = board_center + gauss(0.0, initial_variance)
            momentum = 0.0

            for row_idx in range(num_rows):
                offset = float(row_idx % 2) * half_peg_offset
                peg_virtual_center = position + offset
                delta = (position - peg_virtual_center) * inv_peg_radius
                clipped_delta = max(-1.0, min(1.0, delta))

                bounce_prob = self._calculate_bounce_probability(
                    clipped_delta, bounce_center, bounce_factor, min_prob, max_prob
                )
                direction = 1.0 if random() < bounce_prob else -1.0
                force = (1.0 - abs(clipped_delta)) * elasticity

                momentum = momentum * damping + direction * force * peg_diameter
                position += momentum
                position = max(lower_bound, min(upper_bound, position))

            yield int(position)

    @staticmethod
    @lru_cache(maxsize=BoardConfig.BOUNCE_PROB_CACHE_SIZE)
    def _calculate_bounce_probability(
        delta: float,
        center: float,
        factor: float,
        min_prob: float,
        max_prob: float,
    ) -> float:
        probability = center + factor * delta
        return max(min_prob, min(max_prob, probability))

    def _apply_smoothing(self, slots: Sequence[Frequency]) -> None:
        window_size = BoardConfig.SMOOTHING_WINDOW
        if window_size <= 0:
            self.slot_counts[:] = slots
            return

        slot_length = len(slots)
        if slot_length == 0:
            self.slot_counts = []
            return

        smoothed_counts = [0] * slot_length
        half_window = window_size // 2

        for i in range(slot_length):
            start_idx = max(0, i - half_window)
            end_idx = min(slot_length, i + half_window + 1)
            segment = slots[start_idx:end_idx]
            segment_len = len(segment)
            smoothed_counts[i] = sum(segment) // segment_len if segment_len else 0

        self.slot_counts[:] = smoothed_counts

    def _prepare_drawing_surface(self) -> None:
        if self.image is None or self.draw is None:
            self.image = Image.new(
                "RGB",
                (self.board_width, self.board_height),
                BoardConfig.BACKGROUND_COLOR,
            )
            self.draw = ImageDraw.Draw(self.image)
        else:
            # Optimized clear: draw a single background rectangle
            self.draw.rectangle(
                (0, 0, self.board_width, self.board_height),
                fill=BoardConfig.BACKGROUND_COLOR,
            )


    def generate_image(self) -> Image.Image:
        if not self._simulation_complete:
            raise RuntimeError("Simulation must be run before image generation.")

        self._prepare_drawing_surface()
        assert self.image is not None, "Image should be initialized"
        assert self.draw is not None, "Draw context should be initialized"

        max_frequency = max(self.slot_counts, default=0)
        if max_frequency <= 0:
            logging.info("No positive counts; returning blank image.")
            return self.image

        num_slots = len(self.slot_counts)
        if num_slots == 0:
             logging.warning("Zero slots; returning blank image.")
             return self.image

        bar_width = max(
            BoardConfig.HISTOGRAM_BAR_MIN_WIDTH,
            self.board_width // num_slots,
        )

        self._draw_all_bars(max_frequency, bar_width)
        return self.image

    def _draw_all_bars(self, max_frequency: int, bar_width: int) -> None:
        assert self.draw is not None, "Draw context must exist"
        assert max_frequency > 0, "Max frequency must be positive"

        left_color = BoardConfig.LEFT_COLOR
        right_color = BoardConfig.RIGHT_COLOR
        half_width_pixels = self._half_board_width_pixels
        board_h = float(self.board_height)
        board_w = self.board_width
        inv_max_freq = 1.0 / max_frequency

        for idx, frequency in enumerate(self.slot_counts):
            if frequency <= 0:
                continue

            x_start = idx * bar_width
            if x_start >= board_w:
                continue

            # Calculate height before checking width to avoid redundant calculations
            bar_height = int(frequency * inv_max_freq * board_h)
            if bar_height <= 0:
                continue

            # Ensure bar end is within bounds and width is positive
            x_end = min(x_start + bar_width, board_w)
            if x_end <= x_start:
                continue

            y_start = int(board_h) - bar_height
            color = left_color if x_start < half_width_pixels else right_color

            self.draw.rectangle(
                (x_start, y_start, x_end, int(board_h)), fill=color
            )

    def save_image(self, filename: Optional[str | Path] = None) -> Path:
        try:
            current_image = self.generate_image()
        except RuntimeError as exc:
            logging.error(f"Image generation failed: {exc}")
            raise

        output_path: Path
        if filename:
            output_path = Path(filename).resolve()
        else:
            output_path = generate_unique_filename(
                base_name=BoardConfig.DEFAULT_IMAGE_BASENAME, suffix=".png"
            )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            current_image.save(output_path)
            logging.info(f"Image successfully saved: {output_path}")
            return output_path
        except (IOError, OSError, ValueError) as exc:
            logging.error(f"Failed to save image to {output_path}: {exc}")
            raise
        except Exception as exc:
            logging.exception(f"Unexpected error during image save: {exc}")
            raise


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format=BoardConfig.LOG_FORMAT, force=True
    )


def generate_unique_filename(base_name: str, suffix: str) -> Path:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    filename = f"{base_name}_{timestamp}{suffix}"
    return Path(filename).resolve()


def run_simulation_and_save() -> None:
    try:
        board = GaltonBoard()
        board.simulate()
        saved_path = board.save_image()
        logging.info(f"Process complete. Output: {saved_path}")
    except (ValueError, RuntimeError) as exc:
        logging.error(f"Execution failed due to configuration or state: {exc}")
    except (IOError, OSError) as exc:
         logging.error(f"File system error: {exc}")
    except Exception as exc:
        logging.exception(f"An unexpected critical error occurred: {exc}")


def main() -> None:
    setup_logging()
    run_simulation_and_save()


if __name__ == "__main__":
    main()