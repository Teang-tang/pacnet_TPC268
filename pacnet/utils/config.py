"""
Configuration for PACNet evaluation.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # pacnet's parent = plant_counting
OUTPUT_DIR = str(PROJECT_ROOT / 'results' / 'pacnet')
WEIGHTS_DIR = str(PROJECT_ROOT / 'weights')
TPC_DATA_DIR = str(PROJECT_ROOT / 'TPC-268' / 'TPC-268')
TPC_SPLITS = str(PROJECT_ROOT / 'TPC-268-main' / 'splits')
TPC_ANNO = str(PROJECT_ROOT / 'TPC-268-main' / 'annotations' / 'tpc268_annotations.json')

# Model settings
IMAGE_SIZE = 256
SCORE_FACTOR = 1000.0  # Density map scale factor (matching CLIP-Count convention)
