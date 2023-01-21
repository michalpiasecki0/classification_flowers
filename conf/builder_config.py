import os.path
from pathlib import Path

base_paths = list(Path("/home").glob("**/classification_flowers"))

if len(base_paths) != 1:
    raise Exception(
        f"Ambiguous directory location. Please make sure there is only one classification_flowers folder"
        f"{base_paths}"
    )
else:
    BASE_PATH = base_paths[0]

FLOWERS_PATH = os.path.join(
    BASE_PATH,
    "data",
    "flowers",
)
OUTPUT_PATH = os.path.join(BASE_PATH, "data", "dataset")
TRAIN_TEST_VAL_RATIO = [0.8, 0.1, 0.1]
