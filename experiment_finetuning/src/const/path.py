import pathlib

ACCESS_TOKEN = "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"
PROJECT_ROOT = (pathlib.Path(__file__) / ".." / ".." / "..").resolve()

OUTPUT_DIR = PROJECT_ROOT / "autodl-tmp" / "output"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "model"
LOG_OUTPUT_DIR = OUTPUT_DIR / "log"

DATASET_DIR = PROJECT_ROOT / "autodl-tmp" / "dataset"

CACHE_DIR = PROJECT_ROOT / ".cache"

MIND_SMALL_DATASET_DIR = DATASET_DIR / "small"
MIND_SMALL_VAL_DATASET_DIR = MIND_SMALL_DATASET_DIR / "dev"
MIND_SMALL_TRAIN_DATASET_DIR = MIND_SMALL_DATASET_DIR / "train"
MIND_SMALL_IMG_DATASET_DIR = MIND_SMALL_DATASET_DIR / "IM-MIND"