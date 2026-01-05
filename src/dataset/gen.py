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

- Chenghao Li
"""


import os
import logging
from pathlib import Path

import cv2


from .image import *


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

SYMBOLS_DIR = os.path.join(DIR_PATH, "../../symbols")
BACKGROUNDS_DIR = os.path.join(DIR_PATH, "../../architecture")

DATA_DIR = os.path.join(DIR_PATH, "../../data")
EXTRACTED_IMG_DIR = os.path.join(DATA_DIR, "extracted_arch")
OUTPUT_DIR = os.path.join(DATA_DIR, "dataset")

# CONFIG:
IMAGE_SIZE = Vector2(x=2000, y=2832)

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

# logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Sample(Background):
    bounding_boxes: list[BoundingBox]

    def __post_init__(self):
        self.result: np.array = self.data.copy()
        super().__post_init__()

    def _overlay_symbol(self, symbol: Symbol, position: Vector2) -> None:
        # position is relative, as center
        x = position.x - symbol.dim.x // 2
        y = position.y - symbol.dim.y // 2

        if symbol.data.shape[2] == 4:
            # extract alpha channel
            alpha = symbol.data[:, :, 3] / 255.0
            alpha = alpha[:, :, np.newaxis]

            # blend
            roi = self.result[y:y + symbol.dim.y, x:x + symbol.dim.x]
            blended = (alpha * symbol.data[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
            self.result[y:y + symbol.dim.y, x:x + symbol.dim.x] = blended
        else:
            # simple overlay (with white as transparent)
            gray_symbol = cv2.cvtColor(symbol.data, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_symbol, 250, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)

            roi = self.result[y:y + symbol.dim.y, x:x + symbol.dim.x]
            bg_part = cv2.bitwise_and(roi, roi, mask=mask_inv)
            sym_part = cv2.bitwise_and(symbol.data, symbol.data, mask=mask)
            self.result[y:y + symbol.dim.y, x:x + symbol.dim.x] = cv2.add(bg_part, sym_part)
        
        self.bounding_boxes.append(
            BoundingBox(
                center=self.get_abs(position),
                dim=self.get_abs(symbol.dim),
                symbol=symbol
            )
        )

    


class DatasetGenerator:
    def __init__(self, symbols: list[Symbol], backgrounds: list[Background], output_dir: str, image_size: Vector2=IMAGE_SIZE):
        self.symbols: list[Symbol] = symbols
        self.backgrounds: list[Background] = backgrounds 
        self.output_dir: Path = Path(output_dir)
        self.image_size: Vector2 = image_size
        self.images_dir = Path(os.path.join(self.output_dir, 'images'))
        self.labels_dir = Path(os.path.join(self.output_dir, 'labels'))

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Successfully initialized DatasetGenerator")
    
    @classmethod
    def from_imgs(cls, symbols_dir: str, backgrounds_dir: str, output_dir: str, image_size: Vector2=IMAGE_SIZE):
        # use when you already extracted pdf images
        backgrounds = grab_backgrounds(backgrounds_dir)
        symbols = grab_symbols(symbols_dir, CLASS_IDS, CLASS_PLACEMENT)
        return cls(symbols, backgrounds, output_dir, image_size)

    @classmethod
    def from_pdfs(cls, symbols_dir: str, backgrounds_dir: str, output_dir: str, extract_output: str=EXTRACTED_IMG_DIR, image_size: Vector2=IMAGE_SIZE):
        # use if pdf images aren't extracted
        backgrounds = extract_pdfs(backgrounds_dir, extract_output, image_size)
        symbols = grab_symbols(symbols_dir, CLASS_IDS, CLASS_PLACEMENT)
        return cls(symbols, backgrounds, output_dir, image_size)



def demo_lines(path: str) -> None:
    img = cv2.imread(path)
    lines = detect_lines(img)
    filtered_lines = filter_lines(lines, Vector2.from_shape(img.shape), 0.15)
    for line in filtered_lines:
        cv2.line(img, (line.p1.x, line.p1.y), (line.p2.x, line.p2.y), (0, 0, 255), 2)
    final_img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    cv2.imshow("lines", final_img)
    cv2.waitKey(0)

def main():
    result = DatasetGenerator.from_pdfs(SYMBOLS_DIR, BACKGROUNDS_DIR, OUTPUT_DIR)
    


if __name__ == "__main__":
    main()