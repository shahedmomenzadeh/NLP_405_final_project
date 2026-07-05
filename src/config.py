"""Project-wide configuration constants and paths."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "NLP-prj-data"
FORMAT2_ROOT = DATA_ROOT / "Format2" / "atis"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

BERT_NAME = "bert-base-uncased"

MAX_SEQ_LEN = 64

BATCH_SIZE = 32

SEED = 42

PAD_TAG = "PAD"
PAD_TAG_ID = 0
SUBWORD_TAG = "X"
IGNORE_INDEX = -100
