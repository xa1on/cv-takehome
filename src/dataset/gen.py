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

def extract_pdf_img(path: str, output: str=EXTRACTED_IMG_DIR) -> list[str]:
    """
    get images from a pdf at path
    
    :param path: path of pdf
    :type path: str
    :param output: output dir
    :type path: str
    :return: list of paths of resulting images
    :rtype: list[str]
    """
    result: list[str] = []
    for i, page in enumerate(convert_from_path(path)):
        result_path = os.path.join(output, f"{os.path.basename(path)}-p{i}.png")
        cv2.imwrite(result_path, np.array(page))
        result.append(result_path)
    return result

def extract_pdfs(path: str, output: str=EXTRACTED_IMG_DIR) -> list[str]:
    """
    extract images from all pdfs in path
    
    :param path: path of dir w/ pdfs
    :type path: str
    :param output: output dir
    :type output: str
    :return: list of paths to resulting images
    :rtype: list[str]
    """
    result: list[str] = []
    for file in list(Path(path).glob('*.pdf')):
        result += extract_pdf_img(str(file), output)
    return result



def main():
    extract_pdfs(BACKGROUNDS_DIR)

if __name__ == "__main__":
    main()