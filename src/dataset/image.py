import os
import math
import logging

from enum import Enum
from typing import Self
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from pdf2image import convert_from_path

# LINE DETECTION ARGS
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 175

HOUGH_THRESHOLD = 50
MIN_LINE_LENGTH = 75
MAX_LINE_GAP = 3

GAUSSIAN_BLUR = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlacementMode(Enum):
    """ placement mode for symbols """
    RANDOM = 1
    ON_LINE = 2
    NEXT_TO_LINE = 3

@dataclass
class Vector2:
    """2D vector for positions and dimensions"""
    x: float|int
    y: float|int

    def to_tuple(self) -> tuple[int|float, int|float]:
        """
        convert to tuple (x, y)

        :return: tuple of (x, y)
        :rtype: tuple[int|float, int|float]
        """
        return self.x, self.y

    @classmethod
    def from_list(cls, list: list[int|float] | tuple[int|float]) -> Self:
        """
        create Vector2 from list or tuple

        :param list: list or tuple with [x, y] values
        :type list: list[int|float] | tuple[int|float]
        :return: new Vector2 instance
        :rtype: Self
        """
        return cls(list[0], list[1])

    @classmethod
    def from_shape(cls, dim: list[int|float]) -> Self:
        """
        create Vector2 from numpy shape (height, width) -> (x=width, y=height)

        :param dim: numpy shape array [height, width, ...]
        :type dim: list[int|float]
        :return: new Vector2 with x=width, y=height
        :rtype: Self
        """
        return cls.from_list([dim[1], dim[0]])

@dataclass
class DetectedLine:
    """line detected by Hough transform with computed properties"""
    p1: Vector2
    p2: Vector2

    def __post_init__(self):
        self.angle: float = math.degrees(math.atan2(self.p2.y - self.p1.y, self.p2.x - self.p1.x))
        self.length: float = math.sqrt((self.p2.x - self.p1.x) ** 2 + (self.p2.y - self.p1.y) ** 2)
        self.midpoint: Vector2 = Vector2(x=(self.p1.x + self.p2.x) // 2, y=(self.p1.y + self.p2.y) // 2)

    def point_at_ratio(self, ratio: float) -> Vector2:
        """
        get point along line at given ratio (0=p1, 1=p2)

        :param ratio: position along line (0.0 to 1.0)
        :type ratio: float
        :return: point at the given ratio
        :rtype: Vector2
        """
        x = int(self.p1.x + ratio * (self.p2.x - self.p1.x))
        y = int(self.p1.y + ratio * (self.p2.y - self.p1.y))
        return Vector2(x, y)

    def perpendicular_point(self, base_point: Vector2, distance: float) -> Vector2:
        """
        get point perpendicular to line at given distance from base point

        :param base_point: starting point on the line
        :type base_point: Vector2
        :param distance: distance from base point
        :type distance: float
        :return: point perpendicular to line (randomly on either side)
        :rtype: Vector2
        """
        import random
        perp_angle = math.radians(self.angle + 90)

        if random.random() > 0.5:
            perp_angle += math.pi
        x = int(base_point.x + distance * math.cos(perp_angle))
        y = int(base_point.y + distance * math.sin(perp_angle))
        return Vector2(x, y)

@dataclass
class Image:
    """base image class with dimension utilities"""
    name: str
    data: np.array

    def __post_init__(self):
        self.dim: Vector2 = Vector2.from_shape(self.data.shape)

    def get_rel(self, abs: Vector2) -> Vector2:
        """
        convert normalized coordinates (0-1) to pixel coordinates

        :param abs: normalized position (0-1 range)
        :type abs: Vector2
        :return: pixel coordinates
        :rtype: Vector2
        """
        return Vector2(self.dim.x * abs.x, self.dim.y * abs.y)

    def get_abs(self, rel: Vector2):
        """
        convert pixel coordinates to normalized coordinates (0-1)

        :param rel: pixel position
        :type rel: Vector2
        :return: normalized coordinates
        :rtype: Vector2
        """
        return Vector2(rel.x / self.dim.x, rel.y / self.dim.y)

    def resize(self, new: Vector2) -> None:
        """
        resize image to new dimensions

        :param new: new dimensions (width, height)
        :type new: Vector2
        """
        self.data = cv2.resize(self.data, new.to_tuple())

@dataclass
class Background(Image):
    """image with detected lines for symbol placement"""
    lines: list[DetectedLine]

    @classmethod
    def from_img(cls, img: Image):
        """
        create Background from Image by detecting lines

        :param img: source image
        :type img: Image
        :return: background with detected lines
        :rtype: Background
        """
        return cls(name=img.name, data=img.data, lines=detect_lines(img.data))

@dataclass
class Symbol(Image):
    """symbol image with class id and placement mode"""
    id: int
    placement: PlacementMode

    def transform(self, scale: float=1, rotation: float=0) -> Self:
        """
        create transformed copy of symbol with scale and rotation

        :param scale: scale factor (1.0 = original size)
        :type scale: float
        :param rotation: rotation angle in degrees
        :type rotation: float
        :return: new transformed Symbol
        :rtype: Self
        """
        img = self.data.copy()

        if scale != 1.0:
            new_width = int(img.shape[1] * scale)
            new_height = int(img.shape[0] * scale)
            if new_width > 0 and new_height > 0:
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        if rotation != 0.0:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            cos = abs(rotation_matrix[0, 0])
            sin = abs(rotation_matrix[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            rotation_matrix[0, 2] += (new_w - w) / 2
            rotation_matrix[1, 2] += (new_h - h) / 2

            # fill bg based on alpha channel
            if img.shape[2] == 4:
                img = cv2.warpAffine(img, rotation_matrix, (new_w, new_h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255, 0))
            else:
                img = cv2.warpAffine(img, rotation_matrix, (new_w, new_h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))

        return Symbol(name=self.name, data=img, id=self.id, placement=self.placement)

@dataclass
class BoundingBox:
    """bounding box for symbol with YOLO format export"""
    center: Vector2  # normalized (0-1)
    dim: Vector2  # normalized (0-1)
    symbol: Symbol

    def __post_init__(self):
        self.tl = Vector2(self.center.x - (self.dim.x / 2), self.center.y - (self.dim.y / 2))
        self.br = Vector2(self.center.x + (self.dim.x / 2), self.center.y + (self.dim.y / 2))

    def overlap(self, other: Self) -> bool:
        """
        check if this box overlaps with another

        :param other: other bounding box
        :type other: Self
        :return: True if boxes overlap
        :rtype: bool
        """
        return not (self.tl.x > other.br.x
                    or self.br.x < other.tl.x
                    or self.tl.y > other.br.y
                    or self.br.y < other.tl.y)

    def to_yolo_format(self) -> str:
        """
        convert to YOLO format string: class_id center_x center_y width height

        :return: YOLO format annotation line
        :rtype: str
        """
        return f"{self.symbol.id} {self.center.x:.6f} {self.center.y:.6f} {self.dim.x:.6f} {self.dim.y:.6f}"


def detect_lines(img: np.array) -> list[DetectedLine]:
    """
    detect lines in image using Canny edge detection and Hough transform

    :param img: BGR image array
    :type img: np.array
    :return: list of detected lines
    :rtype: list[DetectedLine]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (GAUSSIAN_BLUR, GAUSSIAN_BLUR), 0)
    edges = cv2.Canny(blurred, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP
    )

    detected_lines: list[DetectedLine] = []
    if not lines is None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines.append(
                DetectedLine(
                    p1=Vector2(x1, y1),
                    p2=Vector2(x2, y2)
                )
            )

    logger.info(f"Detected {len(detected_lines)} lines.")

    return detected_lines



def tile_background(background: Background, tile_size: Vector2, margin: float = 0.15) -> list[Background]:
    """
    tile a background image into smaller chunks, ignoring margins
    :param background: the background image to tile
    :type background: Background
    :param tile_size: the size of each tile
    :type tile_size: Vector2
    :param margin: the percentage of the image to ignore around the edges (0.0 to 1.0)
    :type margin: float
    :return: a list of background tiles
    :rtype: list[Background]
    """
    tiles: list[Background] = []
    img_dim = background.dim
    # define margins
    margin_x = int(img_dim.x * margin)
    margin_y = int(img_dim.y * margin)
    # crop image to the area within margins
    cropped_data = background.data[margin_y:img_dim.y-margin_y, margin_x:img_dim.x-margin_x]
    cropped_dim = Vector2.from_shape(cropped_data.shape)
    for y in range(0, cropped_dim.y, tile_size.y):
        for x in range(0, cropped_dim.x, tile_size.x):
            tile_data = cropped_data[y:y+tile_size.y, x:x+tile_size.x]
            if tile_data.shape[0] != tile_size.y or tile_data.shape[1] != tile_size.x:
                continue
            tile_name = f"{background.name}-tile-{x+margin_x}-{y+margin_y}.png"
            lines = detect_lines(tile_data)
            if lines:
                tile_background = Background(name=tile_name, data=tile_data, lines=lines)
                tiles.append(tile_background)
    logger.info(f"Tiled {background.name} into {len(tiles)} tiles.")
    return tiles

def extract_pdf_img(path: str, output: str, tile_size: Vector2) -> list[Background]:
    """
    extract pages from a single PDF as background images, then tile them

    :param path: path to PDF file
    :type path: str
    :param output: directory to save extracted images
    :type output: str
    :param tile_size: the size of each tile
    :type tile_size: Vector2
    :return: list of Background objects for each tile
    :rtype: list[Background]
    """
    result: list[Background] = []
    for i, page in enumerate(convert_from_path(path)):
        name = f"{os.path.basename(path)}-p{i}"
        data = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        
        # create a single background for the whole page to tile it
        page_background = Background(name=name, data=data, lines=[]) # No need to detect lines on the whole page
        
        # tile the background
        tiles = tile_background(page_background, tile_size)
        
        for tile in tiles:
            result_path = os.path.join(output, tile.name)
            cv2.imwrite(result_path, tile.data)
            result.append(tile)

    logger.info(f"Extracted and tiled {len(result)} background images from {path}.")
    return result


def extract_pdfs(path: str, output: str, tile_size: Vector2) -> list[Background]:
    """
    extract pages from all PDFs in directory as background images

    :param path: directory containing PDF files
    :type path: str
    :param output: directory to save extracted images
    :type output: str
    :return: list of Background objects from all PDFs
    :rtype: list[Background]
    """
    result: list[Background] = []
    for file in list(Path(path).glob('*.pdf')):
        result += extract_pdf_img(str(file), output, tile_size)
    logger.info(f"Extracted {len(result)} background images total from {path}.")
    return result


def extract_images(path: str) -> list[Image]:
    """
    load PNG images from directory as Image objects

    :param path: directory containing PNG files
    :type path: str
    :return: list of Image objects
    :rtype: list[Image]
    """
    result: list[Image] = []
    for file in list(Path(path).glob('*.png')):
        result.append(Image(name=file.stem, data=cv2.imread(str(file))))
    logger.info(f"Extracted {len(result)} images from {path}.")
    return result


def grab_backgrounds(path: str) -> list[Background]:
    """
    load PNG images from directory as Background objects with line detection

    :param path: directory containing PNG files
    :type path: str
    :return: list of Background objects with detected lines
    :rtype: list[Background]
    """
    result: list[Background] = []
    for file in list(Path(path).glob('*.png')):
        data = cv2.imread(str(file))
        result.append(Background(name=file.stem, data=data, lines=detect_lines(data)))
    logger.info(f"Grabbed {len(result)} background images from {path}.")
    return result


def grab_symbols(path: str, ids: dict[str: int], placement: dict[str: PlacementMode]) -> list[Symbol]:
    """
    load symbol PNG images from directory

    :param path: directory containing symbol PNG files
    :type path: str
    :param ids: mapping of symbol name to class id
    :type ids: dict[str: int]
    :param placement: mapping of symbol name to placement mode
    :type placement: dict[str: PlacementMode]
    :return: list of Symbol objects
    :rtype: list[Symbol]
    """
    result: list[Symbol] = []
    for file in list(Path(path).glob('*.png')):
        name = file.stem
        new_symbol = Symbol(name=name, data=cv2.imread(str(file), cv2.IMREAD_UNCHANGED), placement=placement[name], id=ids[name])
        result.append(new_symbol)
        logger.info(f"Loaded symbol {name} from {str(file)}")
    logger.info(f"Grabbed {len(result)} symbols from {path}.")
    return result