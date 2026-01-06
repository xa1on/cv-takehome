"""
Sample class for generating synthetic training samples with symbol placement.
"""

import random
from dataclasses import dataclass

import cv2
import numpy as np

from .image import Background, BoundingBox, Symbol, Vector2, PlacementMode

# Placement configuration constants
MAX_PLACEMENT_ATTEMPTS = 20
MAX_RANDOM_PLACEMENT_ATTEMPTS = 30

# Line filtering thresholds (minimum line length in pixels)
MIN_LINE_LENGTH_ON = 100
MIN_LINE_LENGTH_NEXT_TO = 80

# Position along line (ratio 0-1, avoiding endpoints)
LINE_RATIO_MIN_ON = 0.2
LINE_RATIO_MAX_ON = 0.8
LINE_RATIO_MIN_NEXT_TO = 0.15
LINE_RATIO_MAX_NEXT_TO = 0.85

# Symbol scale ranges (relative to image size)
SCALE_RANGE_ON_LINE = (0.02, 0.10)
SCALE_RANGE_NEXT_TO_LINE = (0.04, 0.10)
SCALE_RANGE_RANDOM = (0.04, 0.10)

# Perpendicular offset range for NEXT_TO_LINE placement (pixels)
PERPENDICULAR_OFFSET_RANGE = (30, 80)

# Margin from image edge for random placement (pixels)
RANDOM_PLACEMENT_MARGIN = 50

# Overlap detection margin (pixels)
OVERLAP_MARGIN = 10

# Visualization constants
BBOX_COLORS = {
    0: (0, 0, 255),    # bowtie - red
    1: (0, 255, 0),    # keynote - green
    2: (255, 0, 0),    # T_Symbol - blue
}
LABEL_FONT_SCALE = 0.5
LABEL_THICKNESS = 1


@dataclass
class Sample(Background):
    def __post_init__(self):
        self.result: np.array = self.data.copy()
        self.bounding_boxes: list[BoundingBox] = []
        super().__post_init__()

    def _is_valid_position(self, position: Vector2, symbol: Symbol, margin: int = OVERLAP_MARGIN) -> bool:
        half_w = symbol.dim.x // 2 + margin
        half_h = symbol.dim.y // 2 + margin

        if (position.x - half_w < 0 or position.x + half_w >= self.dim.x or
            position.y - half_h < 0 or position.y + half_h >= self.dim.y):
            return False

        new_box = BoundingBox(
            center=self.get_abs(position),
            dim=self.get_abs(symbol.dim),
            symbol=symbol
        )
        for existing in self.bounding_boxes:
            if new_box.overlap(existing):
                return False
        return True

    def _overlay_symbol(self, symbol: Symbol, position: Vector2) -> bool:
        x = int(position.x - symbol.dim.x // 2)
        y = int(position.y - symbol.dim.y // 2)

        if x < 0 or y < 0 or x + symbol.dim.x > self.dim.x or y + symbol.dim.y > self.dim.y:
            return False

        if symbol.data.shape[2] == 4:
            # Use alpha channel for blending
            alpha = symbol.data[:, :, 3] / 255.0
            alpha = alpha[:, :, np.newaxis]

            roi = self.result[y:y + symbol.dim.y, x:x + symbol.dim.x]
            blended = (alpha * symbol.data[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
            self.result[y:y + symbol.dim.y, x:x + symbol.dim.x] = blended
        else:
            self.result[y:y + symbol.dim.y, x:x + symbol.dim.x] = symbol.data[:, :, :3]
        self.bounding_boxes.append(
            BoundingBox(
                center=self.get_abs(position),
                dim=self.get_abs(symbol.dim),
                symbol=symbol
            )
        )
        return True

    def place_symbol_on_line(self, symbol: Symbol, scale_range: tuple[float, float] = SCALE_RANGE_ON_LINE) -> bool:
        # place symbol ON a detected line, aligned with line direction
        if not self.lines:
            return False

        good_lines = [l for l in self.lines if l.length > MIN_LINE_LENGTH_ON]
        if not good_lines:
            good_lines = self.lines

        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            line = random.choice(good_lines)
            ratio = random.uniform(LINE_RATIO_MIN_ON, LINE_RATIO_MAX_ON)
            position = line.point_at_ratio(ratio)

            scale = random.uniform(*scale_range)
            transformed = symbol.transform(scale=scale, rotation=-line.angle)

            if self._is_valid_position(position, transformed):
                return self._overlay_symbol(transformed, position)

        return False

    def place_symbol_next_to_line(self, symbol: Symbol, scale_range: tuple[float, float] = SCALE_RANGE_NEXT_TO_LINE, offset_range: tuple[int, int] = PERPENDICULAR_OFFSET_RANGE) -> bool:
        # place symbol NEAR a detected line but not on it
        if not self.lines:
            return False

        good_lines = [l for l in self.lines if l.length > MIN_LINE_LENGTH_NEXT_TO]
        if not good_lines:
            good_lines = self.lines

        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            line = random.choice(good_lines)
            ratio = random.uniform(LINE_RATIO_MIN_NEXT_TO, LINE_RATIO_MAX_NEXT_TO)
            base_point = line.point_at_ratio(ratio)

            offset = random.uniform(*offset_range)
            position = line.perpendicular_point(base_point, offset)

            scale = random.uniform(*scale_range)
            transformed = symbol.transform(scale=scale)

            if self._is_valid_position(position, transformed):
                return self._overlay_symbol(transformed, position)

        return False

    def place_symbol_random(self, symbol: Symbol, scale_range: tuple[float, float] = SCALE_RANGE_RANDOM, margin: int = RANDOM_PLACEMENT_MARGIN) -> bool:
        # Place symbol at random position within bounds
        for _ in range(MAX_RANDOM_PLACEMENT_ATTEMPTS):
            x = random.randint(margin, int(self.dim.x) - margin)
            y = random.randint(margin, int(self.dim.y) - margin)
            position = Vector2(x, y)

            scale = random.uniform(*scale_range)
            transformed = symbol.transform(scale=scale)

            if self._is_valid_position(position, transformed):
                return self._overlay_symbol(transformed, position)

        return False

    def place_symbol(self, symbol: Symbol) -> bool:
        # Place symbol based on placement mode
        if symbol.placement == PlacementMode.ON_LINE:
            return self.place_symbol_on_line(symbol)
        elif symbol.placement == PlacementMode.NEXT_TO_LINE:
            return self.place_symbol_next_to_line(symbol)
        else:
            return self.place_symbol_random(symbol)

    def save(self, image_path: str, label_path: str) -> None:
        # saves to yolo format
        cv2.imwrite(image_path, self.result)
        with open(label_path, 'w') as f:
            for box in self.bounding_boxes:
                f.write(box.to_yolo_format() + '\n')

    def draw_bounding_boxes(self) -> np.array:
        # visaulize bounding boxes of placed signals
        vis = self.result.copy()
        for box in self.bounding_boxes:
            tl = self.get_rel(box.tl)
            br = self.get_rel(box.br)

            color = BBOX_COLORS.get(box.symbol.id, (255, 255, 255))
            cv2.rectangle(vis, (int(tl.x), int(tl.y)), (int(br.x), int(br.y)), color, 2)

            label = box.symbol.name
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS)

            cv2.rectangle(vis, (int(tl.x), int(tl.y) - text_h - 4), (int(tl.x) + text_w, int(tl.y)), color, -1)
            cv2.putText(vis, label, (int(tl.x), int(tl.y) - 2), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, (255, 255, 255), LABEL_THICKNESS)

        return vis
