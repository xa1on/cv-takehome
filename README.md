# CV Take-Home Challenge: Synthetic Dataset Generation & Symbol Detection

## Includes

- `src.dataset.gen` dataset generator
- `src.inference.infer` run inference

## Installation

- Use pip to install dependencies
```sh
pip install -r requirements.txt
```

## Training

- put background pdfs in `architecture`
- put symbols in `symbols`
- run dataset generation script
    - `python -m src.dataset.gen --from-pdfs --num-samples 500`
    - flags (only the important ones):
        - `--num-samples`- number of samples
        - `--from-pdfs`- extract images for training from pdfs
    - you can also edit the constants in the python files under `src.dataset` for more configuration options
- zip the generated dataset folder in `data/`
- use the python notebook in `src.train` (in colab or locally)
- it'll show validation set stats

## Inference

- place the model in `models/`
- run inference script
    - `python -m src.inference.infer .\test_set\PublicTestHVAC.pdf`
- this'll output inference results in `data/inference_results/`
- If you want to run validation, run the cells under validation in `src.train.train_yolo.ipynb`


### Original README / project specs

<details>

## Overview

This challenge tests your ability to create a synthetic dataset generator and train a computer vision model for symbol detection. The goal is to determine whether synthetic training data can effectively train a model to detect symbols in real-world architectural documents.

## Before You Start

Take a moment to look at the 3 symbols in `symbols/` and then examine `test_set/PublicTestHVAC.pdf` to see how these symbols appear in a real architectural document. This will give you useful intuition for designing your synthetic data generator—understanding the scale, context, and visual characteristics of the symbols in practice will help you make better decisions.

## The Task

1. **Build a synthetic dataset generator** that creates training images by:
   - Using the architectural PDFs in `architecture/` as backgrounds (or foregrounds)
   - Randomly overlaying symbols from `symbols/` onto these backgrounds
   - Generating labeled training data in **YOLO format** (images + corresponding `.txt` annotation files with normalized bounding boxes)

2. **Train a CV model** to detect the 3 target symbols in architectural drawings

3. **Validate your model** using the test set provided

## Folder Structure

```
├── symbols/           # The 3 symbols your model should detect
│   ├── bowtie.png
│   ├── keynote.png
│   └── T Symbol.png
├── architecture/      # Background PDFs for synthetic data generation
│   ├── OCM081.1.08.18.pdf
│   ├── PL24.095-Architectural-Plans.pdf
│   └── Planspdf-R.pdf
└── test_set/          # Real-world example showing symbols in context
    └── PublicTestHVAC.pdf
```

## Guidelines

- **test_set**: You can use this for validation. We have a separate held-out test set that we'll use to evaluate your final model.
- **Scope**: Don't worry about perfecting the solution. The goal is to build something functional and demonstrate your approach.
- **What we're looking for**: Your reasoning, design decisions, and ability to explain your work during the follow-up call.

## Deliverables

1. Code for synthetic dataset generation
2. Training pipeline and trained model
3. Brief documentation of your approach
4. Be prepared to walk through your solution and explain your decisions on a call
