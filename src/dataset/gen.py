"""
Generate architectural datasets for symbol detection

TODO:
1. grab images from architectural pdfs
2. add symbols randomly to architectural pdfs, resembling real architectural documents.
    - hough line transform to detect walls/pipes.
    - strategically place T symbols (thermostats) along walls
    - place bowties (valves??) on pipelines
    - Keynotes: probably just good to put them randomly
3. Generate in YOLO format

- Chenghao Li
"""

import os

import numpy as np
import cv2
from pdf2image import convert_from_path

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

SYMBOLS_DIR = os.path.join(DIR_PATH, "../../symbols")
BACKGROUNDS_DIR = os.path.join(DIR_PATH, "../../architecture")

DATA_DIR = os.path.join(DIR_PATH, "../../data")
EXTRACTED_IMG_DIR = os.path.join(DATA_DIR, "extracted_arch")

def get_imgs(path: str, output: str=EXTRACTED_IMG_DIR) -> list[str]:
    """
    get images from a pdf at path
    
    :param path: path of pdf
    :type path: str
    :param output: output path
    :type path: str
    :return: list of paths of resulting images
    :rtype: list[str]
    """
    paths: str = []
    for i, page in enumerate(convert_from_path(path)):
        result_path = os.path.join(output, f"{os.path.basename(path)}-p{i}.png")
        cv2.imwrite(result_path, np.array(page))
        paths.append(result_path)
    return paths

def main():
    get_imgs(os.path.join(BACKGROUNDS_DIR, "OCM081.1.08.18.pdf"))

if __name__ == "__main__":
    main()