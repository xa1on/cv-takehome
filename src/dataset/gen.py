"""
Generate architectural datasets for symbol detection

1. grab images from architectural pdfs

TODO:
2. add symbols randomly to architectural pdfs, resembling real architectural documents.
    - hough line transform to detect walls/pipes.
    - strategically place T symbols (thermostats) along walls
    - place bowties (valves??) on pipelines
    - Keynotes: probably just good to put them randomly
3. Generate in YOLO format

- Chenghao Li
"""

import os
import math
from enum import Enum
from typing import Self
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from pdf2image import convert_from_path

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

SYMBOLS_DIR = os.path.join(DIR_PATH, "../../symbols")
BACKGROUNDS_DIR = os.path.join(DIR_PATH, "../../architecture")

DATA_DIR = os.path.join(DIR_PATH, "../../data")
EXTRACTED_IMG_DIR = os.path.join(DATA_DIR, "extracted_arch")
OUTPUT_DIR = os.path.join(DATA_DIR, "dataset")

# LINE DETECTION ARGS
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 175

HOUGH_THRESHOLD = 50
MIN_LINE_LENGTH = 75
MAX_LINE_GAP = 3

GAUSSIAN_BLUR = 5

class PlacementMode(Enum):
    RANDOM = 1
    ON_LINE = 2
    NEXT_TO_LINE = 3

# need to configure if adding more symbols
CLASS_IDS = {
    'bowtie': 0,
    'keynote': 1,
    'T_Symbol': 2
}

CLASS_PLACEMENT = {
    'bowtie': PlacementMode.ON_LINE,
    'keynote': PlacementMode.RANDOM,
    'T_Symbol': PlacementMode.NEXT_TO_LINE
}


@dataclass
class Vector2:
    x: int
    y: int

@dataclass
class DetectedLine:
    """Represents a line in the image"""
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

@dataclass
class Symbol(Image):
    placement: PlacementMode

    @classmethod
    def from_path(cls, path: str, placement: PlacementMode) -> Self:
        data = cv2.imread(path)
        name = os.path.basename(path)
        return cls(name=name, data=data, placement=placement)



class DatasetGenerator:
    def __init__(self, symbols: list[Symbol], backgrounds: list[Image], output_dir: str, image_size: Vector2=Vector2(x=2000, y=2832)):
        self.symbols: list[Symbol] = symbols
        self.backgrounds: list[Image] = backgrounds 
        self.output_dir: Path = Path(output_dir)
        self.image_size: Vector2 = image_size
        self.images_dir = os.path.join(self.output_dir, 'images')
        self.labels_dir = os.path.join(self.output_dir, 'labels')

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_imgs(cls, symbols_dir: str, backgrounds_dir: str, output_dir: str, image_size: Vector2=Vector2(x=2000, y=2832)):
        backgrounds = extract_images(backgrounds_dir)
        symbols = extract_symbols(symbols_dir)
        return cls(symbols, backgrounds, output_dir, image_size)

    @classmethod
    def from_pdfs(cls, symbols_dir: str, backgrounds_dir: str, output_dir: str, image_size: Vector2=Vector2(x=2000, y=2832)):
        backgrounds = extract_pdfs(backgrounds_dir)
        symbols = extract_symbols(symbols_dir)
        return cls(symbols, backgrounds, output_dir, image_size)

def extract_pdf_img(path: str, output: str=EXTRACTED_IMG_DIR) -> list[Image]:
    """
    get images from a pdf at path
    
    :param path: path of pdf
    :type path: str
    :param output: output dir
    :type path: str
    :return: list of resulting images
    :rtype: list[Image]
    """
    result: list[Image] = []
    for i, page in enumerate(convert_from_path(path)):
        name = f"{os.path.basename(path)}-p{i}.png"
        result.append(
            Image(name=name, data=np.array(page))
        )
        result_path = os.path.join(output, name)
        cv2.imwrite(result_path, np.array(page))
    return result

def extract_pdfs(path: str, output: str=EXTRACTED_IMG_DIR) -> list[Image]:
    """
    extract images from all pdfs in path
    
    :param path: path of dir w/ pdfs
    :type path: str
    :param output: output dir
    :type output: str
    :return: list of paths to resulting images
    :rtype: list[Image]
    """
    result: list[Image] = []
    for file in list(Path(path).glob('*.pdf')):
        result += extract_pdf_img(str(file), output)
    return result

def extract_images(path: str) -> list[Image]:
    result: list[Image] = []
    for file in list(Path(path).glob('*.png')):
        result.append(Image(name=file.stem, data=cv2.imread(str(file))))
    return result

def extract_symbols(path: str) -> list[Symbol]:
    result: list[Symbol] = []
    for file in list(Path(path).glob('*.png')):
        new_symbol = Symbol(name=file.stem, data=cv2.imread(str(file)), placement=CLASS_PLACEMENT[file.stem])
        result.append(new_symbol)
    return result


def detect_lines(img: np.array) -> list[DetectedLine]:
    """
    detect lines in image
    
    :param img: image to detect from
    :type img: np.array
    :return: list of lines detected
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
    for line in lines:
        x1, y1, x2, y2 = line[0]
        detected_lines.append(
            DetectedLine(
                x1=x1, y1=y1, x2=x2, y2=y2
            )
        )

    return detected_lines

def filter_lines(lines: list[DetectedLine], dim: tuple[int, int, int], scale: float) -> list[DetectedLine]:
    """
    filter out lines not within a certain amount of the edge
    
    :param lines: list of lines
    :type lines: list[DetectedLine]
    :param dim: dimension of image
    :type dim: tuple[int, int, int]
    :param scale: scale from 0 - 1 for the threshold of lines ignored
    :type scale: float
    :return: filtered lines
    :rtype: list[DetectedLine]
    """
    result = []
    for line in lines:
        if line.midpoint.x > dim[1] * (scale / 2) and line.midpoint.x < dim[1] * (1 - (scale / 2)) and line.midpoint.y > dim[0] * (scale / 2) and line.midpoint.y < dim[0] * (1 - (scale / 2)):
            result.append(line)
    return result

def main():
    extract_symbols(SYMBOLS_DIR)
    # example line detection
    img = cv2.imread(os.path.join(EXTRACTED_IMG_DIR, "PL24.095-Architectural-Plans.pdf-p1.png"))
    lines = detect_lines(img)
    filtered_lines = filter_lines(lines, img.shape, 0.15)
    for line in filtered_lines:
        cv2.line(img, (line.x1, line.y1), (line.x2, line.y2), (0, 0, 255), 2)
    final_img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
    cv2.imshow("lines", final_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()