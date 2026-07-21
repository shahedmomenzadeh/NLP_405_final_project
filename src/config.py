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

# ---- Phase 3: training defaults ----
BERT_LR = 2e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 4
EARLY_STOP_MIN_DELTA = 1e-4
GRAD_CLIP = 1.0

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"

# ---- Phase 5: LLM (LM Studio) defaults ----
LLM_BASE_URL = "http://172.19.0.1:1111/v1"
LLM_MODEL = "qwen3-vl-4b-instruct"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT = 120.0

# List of LLMs to evaluate in Phase 5.
# Each entry is {"name": <model_id>, "base_url": <openai-compat url>}.
# Edit this list to add or remove models. The CLI flags --model / --models
# override it. Each model must be loaded in LM Studio (or reachable at the
# configured base_url) before running the test for that model.
LLM_MODELS = [
    {"name": "qwen3-vl-4b-instruct", "base_url": LLM_BASE_URL},
]

PHASE5_DIR = PROJECT_ROOT / "phase5"
SUBSET_SIZE = 20
FEWSHOT_K = 3
