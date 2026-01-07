import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Self

import cv2
import numpy as np
from pdf2image import convert_from_path

# tile size must match training configuration
TILE_SIZE = 1024

# margins to ignore at PDF edges (matching dataset generation)
IMG_MARGINS = 0.0

# overlap between tiles for better edge detection
TILE_OVERLAP = 0.15

# NMS parameters for merging detections across tiles
NMS_IOU_THRESHOLD = 0.5

# visualization constants
BBOX_COLORS = {
    0: (0, 0, 255),    # bowtie - red
    1: (0, 255, 0),    # keynote - green
    2: (255, 0, 0),    # T_Symbol - blue
}
CLASS_NAMES = {
    0: 'bowtie',
    1: 'keynote',
    2: 'T_Symbol'
}
LABEL_FONT_SCALE = 0.6
LABEL_THICKNESS = 2
BBOX_THICKNESS = 2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """single detection result with bounding box and confidence"""
    class_id: int
    confidence: float
    x1: int  # top-left x (absolute pixels)
    y1: int  # top-left y (absolute pixels)
    x2: int  # bottom-right x (absolute pixels)
    y2: int  # bottom-right y (absolute pixels)

    @property
    def class_name(self) -> str:
        """get human-readable class name"""
        return CLASS_NAMES.get(self.class_id, f'class_{self.class_id}')

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def iou(self, other: Self) -> float:
        """
        calculate intersection over union with another detection

        :param other: other detection to compare
        :type other: Detection
        :return: IoU value between 0 and 1
        :rtype: float
        """
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        union_area = self.area + other.area - inter_area
        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def to_dict(self) -> dict:
        """convert detection to dictionary format"""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': round(self.confidence, 4),
            'bbox': [self.x1, self.y1, self.x2, self.y2]
        }


@dataclass
class TileInfo:
    """information about a single tile for coordinate mapping"""
    x_offset: int  # tile origin x in full image
    y_offset: int  # tile origin y in full image
    data: np.ndarray  # tile image data


@dataclass
class PageResult:
    """inference results for a single page"""
    page_num: int
    image: np.ndarray
    detections: list[Detection] = field(default_factory=list)

    def draw_detections(self) -> np.ndarray:
        """
        draw bounding boxes and labels on image

        :return: annotated image
        :rtype: np.ndarray
        """
        vis = self.image.copy()
        for det in self.detections:
            color = BBOX_COLORS.get(det.class_id, (255, 255, 255))

            # draw bounding box
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, BBOX_THICKNESS)

            # draw label background and text
            label = f"{det.class_name} {det.confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                   LABEL_FONT_SCALE, LABEL_THICKNESS)

            cv2.rectangle(vis, (det.x1, det.y1 - text_h - 6),
                         (det.x1 + text_w + 4, det.y1), color, -1)
            cv2.putText(vis, label, (det.x1 + 2, det.y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
                       (255, 255, 255), LABEL_THICKNESS)

        return vis

    def summary(self) -> dict:
        """get summary statistics for this page"""
        counts = {}
        for det in self.detections:
            name = det.class_name
            counts[name] = counts.get(name, 0) + 1
        return {
            'page': self.page_num,
            'total_detections': len(self.detections),
            'by_class': counts
        }


class SymbolDetector:
    """tiled inference for symbol detection on architectural documents"""

    def __init__(self, model_path: str, tile_size: int = TILE_SIZE,
                 tile_overlap: float = TILE_OVERLAP, conf_threshold: float = 0.25,
                 nms_threshold: float = NMS_IOU_THRESHOLD, use_onnx: bool = False):
        """
        initialize symbol detector with trained model

        :param model_path: path to trained model (.pt or .onnx)
        :type model_path: str
        :param tile_size: size of tiles for inference (should match training)
        :type tile_size: int
        :param tile_overlap: overlap ratio between adjacent tiles
        :type tile_overlap: float
        :param conf_threshold: minimum confidence threshold for detections
        :type conf_threshold: float
        :param nms_threshold: IoU threshold for non-maximum suppression
        :type nms_threshold: float
        :param use_onnx: whether to use ONNX runtime instead of PyTorch
        :type use_onnx: bool
        """
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.use_onnx = use_onnx

        if use_onnx:
            self._load_onnx_model(model_path)
        else:
            self._load_torch_model(model_path)

        logger.info(f"Initialized SymbolDetector with tile_size={tile_size}, "
                   f"overlap={tile_overlap}, conf={conf_threshold}")

    def _load_torch_model(self, model_path: str) -> None:
        """load PyTorch/ultralytics model"""
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        logger.info(f"Loaded PyTorch model from {model_path}")

    def _load_onnx_model(self, model_path: str) -> None:
        """load ONNX model with DirectML acceleration"""
        import onnxruntime as ort

        # try DirectML first, fall back to CPU
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]

        self.onnx_session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.input_shape = self.onnx_session.get_inputs()[0].shape

        logger.info(f"Loaded ONNX model from {model_path} with providers: {providers}")

    def _create_tiles(self, image: np.ndarray, use_margins: bool = True) -> list[TileInfo]:
        """
        tile image into overlapping chunks for inference

        :param image: full-size image to tile
        :type image: np.ndarray
        :param use_margins: whether to ignore margins like during training
        :type use_margins: bool
        :return: list of tile information with offsets
        :rtype: list[TileInfo]
        """
        h, w = image.shape[:2]
        tiles = []

        # apply margins if requested (matching training data generation)
        if use_margins:
            margin_x = int(w * IMG_MARGINS)
            margin_y = int(h * IMG_MARGINS)
            x_start, x_end = margin_x, w - margin_x
            y_start, y_end = margin_y, h - margin_y
        else:
            x_start, x_end = 0, w
            y_start, y_end = 0, h

        # calculate step size with overlap
        step = int(self.tile_size * (1 - self.tile_overlap))

        for y in range(y_start, y_end, step):
            for x in range(x_start, x_end, step):
                # handle edge cases - extend tile to full size if possible
                tile_x = min(x, x_end - self.tile_size) if x + self.tile_size > x_end else x
                tile_y = min(y, y_end - self.tile_size) if y + self.tile_size > y_end else y

                # skip if tile would be smaller than expected
                if tile_x < x_start or tile_y < y_start:
                    continue

                tile_data = image[tile_y:tile_y + self.tile_size,
                                  tile_x:tile_x + self.tile_size]

                # only include full-size tiles
                if tile_data.shape[0] == self.tile_size and tile_data.shape[1] == self.tile_size:
                    tiles.append(TileInfo(
                        x_offset=tile_x,
                        y_offset=tile_y,
                        data=tile_data
                    ))

        logger.info(f"Created {len(tiles)} tiles from image of size {w}x{h}")
        return tiles

    def _run_inference_torch(self, tile: np.ndarray) -> list[Detection]:
        """run inference on single tile using PyTorch model"""
        results = self.model.predict(tile, conf=self.conf_threshold, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())

                detections.append(Detection(
                    class_id=cls,
                    confidence=conf,
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3])
                ))

        return detections

    def _run_inference_onnx(self, tile: np.ndarray) -> list[Detection]:
        """run inference on single tile using ONNX model"""
        # preprocess: BGR to RGB, HWC to CHW, normalize
        img = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # resize to model input size if needed
        if self.input_shape[2] is not None and self.input_shape[2] != tile.shape[0]:
            # model expects specific size, resize
            target_h, target_w = self.input_shape[2], self.input_shape[3]
            img_resized = cv2.resize(tile, (target_w, target_h))
            img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            scale_x = tile.shape[1] / target_w
            scale_y = tile.shape[0] / target_h
        else:
            scale_x, scale_y = 1.0, 1.0

        outputs = self.onnx_session.run(None, {self.input_name: img})
        predictions = outputs[0]  # shape: (1, num_classes + 4, num_boxes)

        detections = []
        # YOLO output format: (1, 4 + num_classes, num_boxes)
        # transpose to (num_boxes, 4 + num_classes)
        preds = predictions[0].T

        for pred in preds:
            # first 4 values are box coords (cx, cy, w, h), rest are class scores
            cx, cy, w, h = pred[:4]
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.conf_threshold:
                continue

            # convert from center format to corner format
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)

            detections.append(Detection(
                class_id=class_id,
                confidence=confidence,
                x1=x1, y1=y1, x2=x2, y2=y2
            ))

        return detections

    def _run_inference(self, tile: np.ndarray) -> list[Detection]:
        """run inference on single tile"""
        if self.use_onnx:
            return self._run_inference_onnx(tile)
        return self._run_inference_torch(tile)

    def _apply_nms(self, detections: list[Detection]) -> list[Detection]:
        """
        apply non-maximum suppression to remove duplicate detections

        :param detections: list of all detections (may have duplicates)
        :type detections: list[Detection]
        :return: filtered detections after NMS
        :rtype: list[Detection]
        """
        if not detections:
            return []

        # sort by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)

            # remove detections with high IoU to the best detection
            sorted_dets = [d for d in sorted_dets
                          if d.class_id != best.class_id or d.iou(best) < self.nms_threshold]

        logger.info(f"NMS: {len(detections)} -> {len(keep)} detections")
        return keep

    def detect_image(self, image: np.ndarray, use_margins: bool = True) -> list[Detection]:
        """
        run tiled detection on a single image

        :param image: BGR image array
        :type image: np.ndarray
        :param use_margins: whether to apply margins (like training)
        :type use_margins: bool
        :return: list of detections with absolute coordinates
        :rtype: list[Detection]
        """
        tiles = self._create_tiles(image, use_margins=use_margins)
        all_detections = []

        for tile_info in tiles:
            tile_dets = self._run_inference(tile_info.data)

            # translate tile-relative coords to full image coords
            for det in tile_dets:
                det.x1 += tile_info.x_offset
                det.y1 += tile_info.y_offset
                det.x2 += tile_info.x_offset
                det.y2 += tile_info.y_offset
                all_detections.append(det)

        # apply NMS to merge overlapping detections from adjacent tiles
        filtered = self._apply_nms(all_detections)
        return filtered

    def detect_pdf(self, pdf_path: str, use_margins: bool = True) -> list[PageResult]:
        """
        run detection on all pages of a PDF document

        :param pdf_path: path to PDF file
        :type pdf_path: str
        :param use_margins: whether to apply margins during tiling
        :type use_margins: bool
        :return: list of results for each page
        :rtype: list[PageResult]
        """
        logger.info(f"Processing PDF: {pdf_path}")
        pages = convert_from_path(pdf_path)
        results = []

        for i, page in enumerate(pages):
            logger.info(f"Processing page {i + 1}/{len(pages)}")
            image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            detections = self.detect_image(image, use_margins=use_margins)

            results.append(PageResult(
                page_num=i + 1,
                image=image,
                detections=detections
            ))

            logger.info(f"Page {i + 1}: found {len(detections)} detections")

        return results

    def detect_directory(self, input_dir: str, use_margins: bool = True) -> dict[str, list[PageResult]]:
        """
        run detection on all PDFs in a directory

        :param input_dir: directory containing PDF files
        :type input_dir: str
        :param use_margins: whether to apply margins during tiling
        :type use_margins: bool
        :return: dict mapping filename to list of page results
        :rtype: dict[str, list[PageResult]]
        """
        results = {}
        pdf_files = list(Path(input_dir).glob('*.pdf'))
        logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")

        for pdf_path in pdf_files:
            name = pdf_path.name
            results[name] = self.detect_pdf(str(pdf_path), use_margins=use_margins)

        return results


def save_results(results: list[PageResult], output_dir: str, prefix: str = "page") -> None:
    """
    save detection results as annotated images and JSON

    :param results: list of page results to save
    :type results: list[PageResult]
    :param output_dir: directory to save results
    :type output_dir: str
    :param prefix: filename prefix for output files
    :type prefix: str
    """
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for result in results:
        # save annotated image
        annotated = result.draw_detections()
        img_path = output_path / f"{prefix}_{result.page_num:03d}.jpg"
        cv2.imwrite(str(img_path), annotated)

        # collect summary
        summary = result.summary()
        summary['detections'] = [d.to_dict() for d in result.detections]
        all_summaries.append(summary)

        logger.info(f"Saved {img_path}")

    # save JSON summary
    json_path = output_path / f"{prefix}_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    logger.info(f"Saved results summary to {json_path}")


def main():
    """command line interface for inference"""
    import argparse

    parser = argparse.ArgumentParser(description='Run symbol detection on architectural documents')
    parser.add_argument('input', type=str, help='Path to PDF file or directory')
    parser.add_argument('--model', type=str, default='models/best.pt',
                       help='Path to trained model (.pt or .onnx)')
    parser.add_argument('--output', type=str, default='data/inference_results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--nms', type=float, default=0.5,
                       help='NMS IoU threshold (0-1)')
    parser.add_argument('--tile-size', type=int, default=TILE_SIZE,
                       help='Tile size for inference')
    parser.add_argument('--no-margins', action='store_true',
                       help='Disable margin cropping')
    parser.add_argument('--onnx', action='store_true',
                       help='Use ONNX runtime instead of PyTorch')

    args = parser.parse_args()

    # initialize detector
    detector = SymbolDetector(
        model_path=args.model,
        tile_size=args.tile_size,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        use_onnx=args.onnx
    )

    input_path = Path(args.input)
    use_margins = not args.no_margins

    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        # single PDF file
        results = detector.detect_pdf(str(input_path), use_margins=use_margins)
        save_results(results, args.output, prefix=input_path.stem)

    elif input_path.is_dir():
        # directory of PDFs
        all_results = detector.detect_directory(str(input_path), use_margins=use_margins)
        for name, results in all_results.items():
            pdf_output = os.path.join(args.output, Path(name).stem)
            save_results(results, pdf_output, prefix="page")
    else:
        logger.error(f"Invalid input: {args.input}")
        return

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
