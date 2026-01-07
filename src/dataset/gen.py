"""
Generate architectural datasets for symbol detection

1. grab images from architectural pdfs
2. hough line transform to detect walls/pipes
3. add symbols to architectural pdfs based on placement mode:
    - bowties (valves): ON lines (pipelines)
    - T symbols (thermostats): NEXT TO lines (walls)
    - keynotes: RANDOM placement
4. Generate in YOLO format

- Chenghao Li
"""

import os
import random
import shutil
import logging
from pathlib import Path

import cv2

from .image import (
    Vector2, PlacementMode, Background, Symbol,
    detect_lines, extract_pdfs, grab_backgrounds, grab_symbols, grab_hard_negatives
)
from .sample import Sample


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

SYMBOLS_DIR = os.path.join(DIR_PATH, "../../symbols")
HARD_NEGATIVES_DIR = os.path.join(SYMBOLS_DIR, "negatives")
BACKGROUNDS_DIR = os.path.join(DIR_PATH, "../../architecture")

DATA_DIR = os.path.join(DIR_PATH, "../../data")
EXTRACTED_IMG_DIR = os.path.join(DATA_DIR, "extracted_arch")
OUTPUT_DIR = os.path.join(DATA_DIR, "dataset")

TILE_SIZE = Vector2(x=1024, y=1024)

# Symbol class definitions (add new symbols here)
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

# Dataset generation parameters
DEFAULT_NUM_SAMPLES = 100
DEFAULT_TRAIN_SPLIT = 0.8
LOG_INTERVAL = 10

# Symbol count ranges per sample
NUM_ON_LINE_RANGE = (0, 2)
NUM_NEXT_TO_LINE_RANGE = (0, 2)
NUM_RANDOM_RANGE = (0, 2)
NUM_HARD_NEGATIVES_RANGE = (0, 3)

# Demo visualization parameters
DEMO_SYMBOLS_PER_TYPE = (2, 5)
DEMO_DISPLAY_SCALE = 0.5
LINE_FILTER_SCALE = 0.15

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """generates synthetic YOLO datasets by placing symbols on architectural backgrounds"""

    def __init__(self, symbols: list[Symbol], backgrounds: list[Background], output_dir: str,
                 hard_negatives: list[Symbol] = None):
        """
        initalize DatasetGenerator

        :param self:
        :param symbols: list of symbols
        :type symbols: list[Symbol]
        :param backgrounds: list of backgrounds
        :type backgrounds: list[Background]
        :param output_dir: output directory of the dataset
        :type output_dir: str
        :param hard_negatives: list of hard negative symbols
        :type hard_negatives: list[Symbol]
        """
        self.symbols = symbols
        self.backgrounds = backgrounds
        self.hard_negatives = hard_negatives or []
        self.output_dir = Path(output_dir)

        if self.output_dir.exists() and self.output_dir.is_dir():
            logger.info(f"Removing existing dataset at {self.output_dir}")
            shutil.rmtree(self.output_dir)

        self.images_dir = Path(os.path.join(self.output_dir, 'images'))
        self.labels_dir = Path(os.path.join(self.output_dir, 'labels'))

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Successfully initialized DatasetGenerator")

    @classmethod
    def from_imgs(cls, symbols_dir: str, backgrounds_dir: str, output_dir: str,
                  hard_negatives_dir: str = HARD_NEGATIVES_DIR):
        """
        create generator from existing PNG background images

        :param symbols_dir: directory containing symbol PNG files
        :type symbols_dir: str
        :param backgrounds_dir: directory containing background PNG files
        :type backgrounds_dir: str
        :param output_dir: output directory for generated dataset
        :type output_dir: str
        :param hard_negatives_dir: directory containing hard negative PNG files
        :type hard_negatives_dir: str
        :return: initialized DatasetGenerator
        :rtype: DatasetGenerator
        """
        backgrounds = grab_backgrounds(backgrounds_dir)
        symbols = grab_symbols(symbols_dir, CLASS_IDS, CLASS_PLACEMENT)
        hard_negatives = grab_hard_negatives(hard_negatives_dir)
        return cls(symbols, backgrounds, output_dir, hard_negatives)

    @classmethod
    def from_pdfs(cls, symbols_dir: str, backgrounds_dir: str, output_dir: str,
                  extract_output: str = EXTRACTED_IMG_DIR, tile_size: Vector2 = TILE_SIZE,
                  hard_negatives_dir: str = HARD_NEGATIVES_DIR):
        """
        create generator by extracting backgrounds from PDF files

        :param symbols_dir: directory containing symbol PNG files
        :type symbols_dir: str
        :param backgrounds_dir: directory containing PDF files to extract
        :type backgrounds_dir: str
        :param output_dir: output directory for generated dataset
        :type output_dir: str
        :param extract_output: directory to save extracted PDF images
        :type extract_output: str
        :param tile_size: the size of each tile
        :type tile_size: Vector2
        :param hard_negatives_dir: directory containing hard negative PNG files
        :type hard_negatives_dir: str
        :return: initialized DatasetGenerator
        :rtype: DatasetGenerator
        """
        Path(extract_output).mkdir(parents=True, exist_ok=True)
        backgrounds = extract_pdfs(backgrounds_dir, extract_output, tile_size)
        symbols = grab_symbols(symbols_dir, CLASS_IDS, CLASS_PLACEMENT)
        hard_negatives = grab_hard_negatives(hard_negatives_dir)
        return cls(symbols, backgrounds, output_dir, hard_negatives)

    def _get_symbol_by_placement(self, placement: PlacementMode):
        """
        return a random symbol matching the given placement mode

        :param placement: the placement mode to filter by
        :type placement: PlacementMode
        :return: random matching symbol or None if no match
        :rtype: Symbol | None
        """
        matching = [s for s in self.symbols if s.placement == placement]
        return random.choice(matching) if matching else None

    def generate_sample(self, background: Background,
                        num_on_line: tuple[int, int] = NUM_ON_LINE_RANGE,
                        num_next_to_line: tuple[int, int] = NUM_NEXT_TO_LINE_RANGE,
                        num_random: tuple[int, int] = NUM_RANDOM_RANGE,
                        num_hard_negatives: tuple[int, int] = NUM_HARD_NEGATIVES_RANGE) -> Sample:
        """
        generate a single sample with symbols placed on background

        :param background: the background image to place symbols on
        :type background: Background
        :param num_on_line: range (min, max) of ON_LINE symbols to place
        :type num_on_line: tuple[int, int]
        :param num_next_to_line: range (min, max) of NEXT_TO_LINE symbols to place
        :type num_next_to_line: tuple[int, int]
        :param num_random: range (min, max) of RANDOM symbols to place
        :type num_random: tuple[int, int]
        :param num_hard_negatives: range (min, max) of hard negatives to place
        :type num_hard_negatives: tuple[int, int]
        :return: generated sample with placed symbols
        :rtype: Sample
        """
        sample = Sample(name=background.name, data=background.data.copy(), lines=background.lines)

        for placement, count_range in [
            (PlacementMode.ON_LINE, num_on_line),
            (PlacementMode.NEXT_TO_LINE, num_next_to_line),
            (PlacementMode.RANDOM, num_random)
        ]:
            symbol = self._get_symbol_by_placement(placement)
            if symbol:
                n = random.randint(*count_range)
                placed = sum(1 for _ in range(n) if sample.place_symbol(symbol))
                logger.debug(f"Placed {placed}/{n} {placement.name} symbols")

        # place hard negatives
        if self.hard_negatives:
            n = random.randint(*num_hard_negatives)
            for _ in range(n):
                negative = random.choice(self.hard_negatives)
                sample.place_symbol(negative)

        return sample

    def generate_dataset(self, num_samples: int = DEFAULT_NUM_SAMPLES, train_split: float = DEFAULT_TRAIN_SPLIT, apply_noise: bool = True) -> dict[str, int]:
        """
        generate full dataset with train/val split

        :param num_samples: total number of samples to generate
        :type num_samples: int
        :param train_split: fraction of samples for training (0-1)
        :type train_split: float
        :param apply_noise: whether to apply noise augmentation to samples
        :type apply_noise: bool
        :return: statistics dict with train, val, and total_symbols counts
        :rtype: dict[str, int]
        """
        for split in ['train', 'val']:
            (self.images_dir / split).mkdir(exist_ok=True)
            (self.labels_dir / split).mkdir(exist_ok=True)

        num_train = int(num_samples * train_split)
        stats = {'train': 0, 'val': 0, 'total_symbols': 0}

        for i in range(num_samples):
            split = 'train' if i < num_train else 'val'
            background = random.choice(self.backgrounds)
            sample = self.generate_sample(background)

            image_path = str(self.images_dir / split / f"synthetic_{i:05d}.jpg")
            label_path = str(self.labels_dir / split / f"synthetic_{i:05d}.txt")
            sample.save(image_path, label_path, apply_noise=apply_noise)

            stats[split] += 1
            stats['total_symbols'] += len(sample.bounding_boxes)

            if (i + 1) % LOG_INTERVAL == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples")

        self._generate_yaml()
        logger.info(f"Dataset generation complete!")
        logger.info(f"Train samples: {stats['train']}, Val samples: {stats['val']}")
        logger.info(f"Total symbols placed: {stats['total_symbols']}")

        return stats

    def _generate_yaml(self) -> None:
        names_lines = '\n'.join(f'  {id}: {name}' for name, id in sorted(CLASS_IDS.items(), key=lambda x: x[1]))

        yaml_content = f"""# Synthetic Dataset for Symbol Detection
# Auto-generated by gen.py

path: {self.output_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
{names_lines}

# Number of classes
nc: {len(CLASS_IDS)}
"""
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        logger.info(f"Generated dataset.yaml at {yaml_path}")


def demo_lines(path: str) -> None:
    """
    visualize detected lines on an image

    :param path: path to the image file
    :type path: str
    """
    img = cv2.imread(path)
    lines = detect_lines(img)
    for line in lines:
        cv2.line(img, (line.p1.x, line.p1.y), (line.p2.x, line.p2.y), (0, 0, 255), 2)
    final_img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    cv2.imshow("lines", final_img)
    cv2.waitKey(0)


def demo_placement(background_path: str = None, save_path: str = None) -> None:
    """
    demo symbol placement with bounding box visualization

    :param background_path: optional path to specific background image
    :type background_path: str | None
    :param save_path: optional path to save visualization
    :type save_path: str | None
    """
    symbols = grab_symbols(SYMBOLS_DIR, CLASS_IDS, CLASS_PLACEMENT)

    if background_path:
        data = cv2.imread(background_path)
        name = os.path.basename(background_path)
        lines = detect_lines(data)
        background = Background(name=name, data=data, lines=lines)
    else:
        if os.path.exists(EXTRACTED_IMG_DIR) and list(Path(EXTRACTED_IMG_DIR).glob('*.png')):
            backgrounds = grab_backgrounds(EXTRACTED_IMG_DIR)
        else:
            Path(EXTRACTED_IMG_DIR).mkdir(parents=True, exist_ok=True)
            backgrounds = extract_pdfs(BACKGROUNDS_DIR, EXTRACTED_IMG_DIR, TILE_SIZE)
        background = random.choice(backgrounds)

    sample = Sample(name=background.name, data=background.data.copy(), lines=background.lines)

    for symbol in symbols:
        for _ in range(random.randint(*DEMO_SYMBOLS_PER_TYPE)):
            sample.place_symbol(symbol)

    vis = sample.draw_bounding_boxes()
    display = cv2.resize(vis, (int(vis.shape[1] * DEMO_DISPLAY_SCALE), int(vis.shape[0] * DEMO_DISPLAY_SCALE)))

    if save_path:
        cv2.imwrite(save_path, vis)
        logger.info(f"Saved demo visualization to {save_path}")

    cv2.imshow("Symbol Placement Demo", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # cli for easy use
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic training dataset')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--from-pdfs', action='store_true', help='Extract images from PDFs (default: use existing images)')
    parser.add_argument('--demo-lines', type=str, help='Path to image for line detection demo')
    parser.add_argument('--demo-placement', action='store_true', help='Demo symbol placement with bounding boxes')
    parser.add_argument('--demo-bg', type=str, help='Background image for demo-placement (optional)')
    parser.add_argument('--demo-save', type=str, help='Save path for demo visualization (optional)')
    parser.add_argument('--no-noise', action='store_true', help='Disable noise augmentation')

    args = parser.parse_args()

    if args.demo_lines:
        demo_lines(args.demo_lines)
        return

    if args.demo_placement:
        demo_placement(background_path=args.demo_bg, save_path=args.demo_save)
        return

    if args.from_pdfs:
        generator = DatasetGenerator.from_pdfs(SYMBOLS_DIR, BACKGROUNDS_DIR, OUTPUT_DIR)
    else:
        generator = DatasetGenerator.from_imgs(SYMBOLS_DIR, EXTRACTED_IMG_DIR, OUTPUT_DIR)

    stats = generator.generate_dataset(num_samples=args.num_samples, apply_noise=not args.no_noise)

    print("Finished!")
    print(f"Train samples: {stats['train']}")
    print(f"Val samples: {stats['val']}")
    print(f"Total symbols: {stats['total_symbols']}")



if __name__ == "__main__":
    main()
