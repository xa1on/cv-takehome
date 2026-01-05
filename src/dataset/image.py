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
    RANDOM = 1
    ON_LINE = 2
    NEXT_TO_LINE = 3

@dataclass
class Vector2:
    x: int
    y: int

    def to_tuple(self) -> tuple[int, int]:
        return self.x, self.y

@dataclass
class DetectedLine:
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self):
        self.angle: float = math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))
        self.length: float = math.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        self.midpoint: Vector2 = Vector2(x=(self.x1 + self.x2) // 2, y=(self.y1 + self.y2) // 2)

@dataclass
class Image:
    name: str
    data: np.array

    def resize(self, new: Vector2) -> None:
        self.data = cv2.resize(self.data, new.to_tuple())

@dataclass
class Background(Image):
    lines: DetectedLine

    @classmethod
    def from_img(cls, img: Image):
        cls(name=img.name, data=img.data, lines=detect_lines(img.data))

@dataclass
class Symbol(Image):
    placement: PlacementMode

    @classmethod
    def from_path(cls, path: str, placement: PlacementMode) -> Self:
        data = cv2.imread(path)
        name = os.path.basename(path)
        return cls(name=name, data=data, placement=placement)
    
    def transform(self, scale: float=1, rotation: float=0) -> Self:
        # copy of symbol w/ given scale and rotation
        img = self.data.copy()

        if scale != 1.0:
            new_width = int(img.shape[1] * scale)
            new_height = int(img.shape[0] * scale)
            if new_width > 0 and new_height > 0:
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
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
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255, 0))
            else:
                img = cv2.warpAffine(img, rotation_matrix, (new_w, new_h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))

        return Symbol(name=self.name, data=img, placement=self.placement)
    

def detect_lines(img: np.array) -> list[DetectedLine]:
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
    for line in lines:
        x1, y1, x2, y2 = line[0]
        detected_lines.append(
            DetectedLine(
                x1=x1, y1=y1, x2=x2, y2=y2
            )
        )

    logger.info(f"Detected {len(detected_lines)} lines.")

    return detected_lines

def filter_lines(lines: list[DetectedLine], dim: tuple[int, int, int], scale: float) -> list[DetectedLine]:
    result = []
    for line in lines:
        if line.midpoint.x > dim[1] * (scale / 2) and line.midpoint.x < dim[1] * (1 - (scale / 2)) and line.midpoint.y > dim[0] * (scale / 2) and line.midpoint.y < dim[0] * (1 - (scale / 2)):
            result.append(line)
    logger.info(f"Filtered to keep {len(result)} lines.")
    return result

def extract_pdf_img(path: str, output: str, image_size: Vector2) -> list[Background]:
    result: list[Background] = []
    for i, page in enumerate(convert_from_path(path)):
        name = f"{os.path.basename(path)}-p{i}.png"
        data = np.array(page)
        data = cv2.resize(data, image_size.to_tuple())
        result.append(
            Background(name=name, data=data, lines=detect_lines(data))
        )
        result_path = os.path.join(output, name)
        cv2.imwrite(result_path, data)
    logger.info(f"Extracted {len(result)} background images from {path}.")
    return result

def extract_pdfs(path: str, output: str, image_size: Vector2) -> list[Background]:
    result: list[Background] = []
    for file in list(Path(path).glob('*.pdf')):
        result += extract_pdf_img(str(file), output, image_size)
    logger.info(f"Extracted {len(result)} background images total from {path}.")
    return result

# some of this code is kind of repeated, but wtv
def extract_images(path: str) -> list[Image]:
    result: list[Image] = []
    for file in list(Path(path).glob('*.png')):
        result.append(Image(name=file.stem, data=cv2.imread(str(file))))
    logger.info(f"Extracted {len(result)} images from {path}.")
    return result

def grab_backgrounds(path: str) -> list[Background]:
    result: list[Background] = []
    for file in list(Path(path).glob('*.png')):
        data = cv2.imread(str(file))
        result.append(Background(name=file.stem, data=data, lines=detect_lines(data)))
    logger.info(f"Grabbed {len(result)} background images from {path}.")
    return result

def grab_symbols(path: str, placement: dict[str: PlacementMode]) -> list[Symbol]:
    result: list[Symbol] = []
    for file in list(Path(path).glob('*.png')):
        name = file.stem
        new_symbol = Symbol(name=name, data=cv2.imread(str(file)), placement=placement[file.stem])
        result.append(new_symbol)
        logger.info(f"Loaded symbol {name} from {str(file)}")
    logger.info(f"Grabbed {len(result)} symbols from {path}.")
    return result