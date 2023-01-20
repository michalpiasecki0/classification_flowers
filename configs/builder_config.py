import os.path
from pathlib import Path

BASE_PATH = list(Path('/home').glob('**/classification_deeptale'))

if len(BASE_PATH) != 1:
    raise Exception(f'Ambiguous directory location. Please make sure there is only one classification_deeptale folder'
                    f'{BASE_PATH}')
else:
    BASE_PATH = BASE_PATH[0]

FLOWERS_PATH = os.path.join(BASE_PATH, 'data', 'flowers', )
OUTPUT_PATH = os.path.join(BASE_PATH, 'data', 'dataset')
TRAIN_TEST_VAL_RATIO = [0.8, 0.1, 0.1]
