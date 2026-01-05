"""
Generate architectural datasets for symbol detection

1. grab images from architectural pdfs
2. hough line transform to detect walls/pipes

TODO:
2. add symbols randomly to architectural pdfs, resembling real architectural documents.
    - strategically place T symbols (thermostats) along walls
    - place bowties (valves??) on pipelines
    - Keynotes: probably just good to put them randomly
3. Generate in YOLO format
4. write documentation :P
5. split into more files, shouldn't clup everything in one file
6. (if i have extra time) yaml config file to setup hough params and general config stuff 

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

@dataclass
class Vector2:
    x: int
    y: int

    def to_tuple(self) -> tuple[int, int]:
        return self.x, self.y

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

SYMBOLS_DIR = os.path.join(DIR_PATH, "../../symbols")
BACKGROUNDS_DIR = os.path.join(DIR_PATH, "../../architecture")

DATA_DIR = os.path.join(DIR_PATH, "../../data")
EXTRACTED_IMG_DIR = os.path.join(DATA_DIR, "extracted_arch")
OUTPUT_DIR = os.path.join(DATA_DIR, "dataset")

# CONFIG:
IMAGE_SIZE = Vector2(x=2000, y=2832)

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
        

class DatasetGenerator:
    def __init__(self, symbols: list[Symbol], backgrounds: list[Background], output_dir: str, image_size: Vector2=IMAGE_SIZE):
        self.symbols: list[Symbol] = symbols
        self.backgrounds: list[Background] = backgrounds 
        self.output_dir: Path = Path(output_dir)
        self.image_size: Vector2 = image_size
        self.images_dir = os.path.join(self.output_dir, 'images')
        self.labels_dir = os.path.join(self.output_dir, 'labels')

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_imgs(cls, symbols_dir: str, backgrounds_dir: str, output_dir: str, image_size: Vector2=IMAGE_SIZE):
        # use when you already extracted pdf images
        backgrounds = extract_backgrounds(backgrounds_dir)
        symbols = extract_symbols(symbols_dir)
        return cls(symbols, backgrounds, output_dir, image_size)

    @classmethod
    def from_pdfs(cls, symbols_dir: str, backgrounds_dir: str, output_dir: str, image_size: Vector2=IMAGE_SIZE):
        # use if pdf images aren't extracted
        backgrounds = extract_pdfs(backgrounds_dir, image_size=image_size)
        symbols = extract_symbols(symbols_dir)
        return cls(symbols, backgrounds, output_dir, image_size)



def extract_pdf_img(path: str, output: str=EXTRACTED_IMG_DIR, image_size: Vector2=IMAGE_SIZE) -> list[Background]:
    result: list[Background] = []
    for i, page in enumerate(convert_from_path(path)):
        name = f"{os.path.basename(path)}-p{i}.png"
        data = np.array(page)
        data = cv2.resize(data, image_size.to_tuple())
        result.append(
            Background(name=name, data=data, lines=detect_lines(data))
        )
        result_path = os.path.join(output, name)
        cv2.imwrite(result_path, np.array(page))
    return result



def extract_pdfs(path: str, output: str=EXTRACTED_IMG_DIR, image_size: Vector2=IMAGE_SIZE) -> list[Background]:
    result: list[Background] = []
    for file in list(Path(path).glob('*.pdf')):
        result += extract_pdf_img(str(file), output, image_size)
    return result

# some of this code is kind of repeated, but wtv
def extract_images(path: str) -> list[Image]:
    result: list[Image] = []
    for file in list(Path(path).glob('*.png')):
        result.append(Image(name=file.stem, data=cv2.imread(str(file))))
    return result

def extract_backgrounds(path: str) -> list[Image]:
    result: list[Background] = []
    for file in list(Path(path).glob('*.png')):
        data = cv2.imread(str(file))
        result.append(Background(name=file.stem, data=data, lines=detect_lines(data)))
    return result

def extract_symbols(path: str) -> list[Symbol]:
    result: list[Symbol] = []
    for file in list(Path(path).glob('*.png')):
        new_symbol = Symbol(name=file.stem, data=cv2.imread(str(file)), placement=CLASS_PLACEMENT[file.stem])
        result.append(new_symbol)
    return result


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

    return detected_lines

def filter_lines(lines: list[DetectedLine], dim: tuple[int, int, int], scale: float) -> list[DetectedLine]:
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