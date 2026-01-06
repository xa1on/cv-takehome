"""
Sample class for generating synthetic training samples with symbol placement.
"""

import random
from dataclasses import dataclass

import cv2
import numpy as np

from .image import Background, BoundingBox, Symbol, Vector2, PlacementMode

# placement configuration constants
MAX_PLACEMENT_ATTEMPTS = 20
MAX_RANDOM_PLACEMENT_ATTEMPTS = 30

# line filtering thresholds (minimum line length in pixels)
MIN_LINE_LENGTH_ON = 100
MIN_LINE_LENGTH_NEXT_TO = 80

# position along line (ratio 0-1, avoiding endpoints)
LINE_RATIO_MIN_ON = 0.2
LINE_RATIO_MAX_ON = 0.8
LINE_RATIO_MIN_NEXT_TO = 0.15
LINE_RATIO_MAX_NEXT_TO = 0.85

# symbol scale ranges (relative to image size)
SCALE_RANGE_ON_LINE = (0.02, 0.10)
SCALE_RANGE_NEXT_TO_LINE = (0.04, 0.10)
SCALE_RANGE_RANDOM = (0.04, 0.10)

# perpendicular offset range for NEXT_TO_LINE placement
PERPENDICULAR_OFFSET_RANGE = (30, 80)

# margin from image edge for random placement
RANDOM_PLACEMENT_MARGIN = 50

# overlap detection margin
OVERLAP_MARGIN = 10

# visualization constants
BBOX_COLORS = {
    0: (0, 0, 255),    # bowtie - red
    1: (0, 255, 0),    # keynote - green
    2: (255, 0, 0),    # T_Symbol - blue
}
LABEL_FONT_SCALE = 0.5
LABEL_THICKNESS = 1


@dataclass
class Sample(Background):
    """synthetic training sample with symbol placement and YOLO export"""

    def __post_init__(self):
        self.result: np.array = self.data.copy()
        self.bounding_boxes: list[BoundingBox] = []
        super().__post_init__()

    def _is_valid_position(self, position: Vector2, symbol: Symbol, margin: int = OVERLAP_MARGIN) -> bool:
        """
        check if position is valid (within bounds and no overlap)

        :param position: pixel position to check
        :type position: Vector2
        :param symbol: symbol to place
        :type symbol: Symbol
        :param margin: extra margin around symbol for overlap detection
        :type margin: int
        :return: True if position is valid
        :rtype: bool
        """
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
        """
        overlay symbol onto result image and record bounding box

        :param symbol: symbol to overlay
        :type symbol: Symbol
        :param position: center position in pixels
        :type position: Vector2
        :return: True if successful
        :rtype: bool
        """
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
        """
        place symbol ON a detected line, aligned with line direction

        :param symbol: symbol to place
        :type symbol: Symbol
        :param scale_range: (min, max) scale factor range
        :type scale_range: tuple[float, float]
        :return: True if placement successful
        :rtype: bool
        """
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
        """
        place symbol NEAR a detected line but not on it

        :param symbol: symbol to place
        :type symbol: Symbol
        :param scale_range: (min, max) scale factor range
        :type scale_range: tuple[float, float]
        :param offset_range: (min, max) perpendicular offset in pixels
        :type offset_range: tuple[int, int]
        :return: True if placement successful
        :rtype: bool
        """
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
        """
        place symbol at random position within bounds

        :param symbol: symbol to place
        :type symbol: Symbol
        :param scale_range: (min, max) scale factor range
        :type scale_range: tuple[float, float]
        :param margin: minimum distance from image edges in pixels
        :type margin: int
        :return: True if placement successful
        :rtype: bool
        """
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
        """
        place symbol based on its placement mode

        :param symbol: symbol to place
        :type symbol: Symbol
        :return: True if placement successful
        :rtype: bool
        """
        if symbol.placement == PlacementMode.ON_LINE:
            return self.place_symbol_on_line(symbol)
        elif symbol.placement == PlacementMode.NEXT_TO_LINE:
            return self.place_symbol_next_to_line(symbol)
        else:
            return self.place_symbol_random(symbol)

    def save(self, image_path: str, label_path: str) -> None:
        """
        save sample image and YOLO format labels

        :param image_path: path to save image
        :type image_path: str
        :param label_path: path to save label file
        :type label_path: str
        """
        cv2.imwrite(image_path, self.result)
        with open(label_path, 'w') as f:
            for box in self.bounding_boxes:
                f.write(box.to_yolo_format() + '\n')

    def draw_bounding_boxes(self) -> np.array:
        """
        draw bounding boxes on image for visualization

        :return: image with drawn bounding boxes and labels
        :rtype: np.array
        """
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
