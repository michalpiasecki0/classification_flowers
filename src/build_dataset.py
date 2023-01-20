"""DESCRIPTION"""

from typing import List
from pathlib import Path
from imutils import paths
import shutil
import numpy as np

import conf.builder_config as config

def copy_images(image_paths: List[str],
                folder: Path
                ):

    if folder.exists():
        print(f'{folder} already existed, it was removed and new content is placed')
        shutil.rmtree(str(folder.absolute()))
    folder.mkdir(parents=True)

    for path in image_paths:
        label, name = path.split(sep='/')[-2:]

        if not (folder / label).exists():
            (folder / label).mkdir(parents=True)
        shutil.copy(path, str((folder / label / name).absolute()))


if __name__ == '__main__':
    image_paths = list(paths.list_images(config.FLOWERS_PATH))
    np.random.shuffle(image_paths)

    train_last_index = int(len(image_paths) * config.TRAIN_TEST_VAL_RATIO[0])
    val_last_index = train_last_index + int(len(image_paths) * config.TRAIN_TEST_VAL_RATIO[1])

    train_paths = image_paths[:train_last_index]
    val_paths = image_paths[train_last_index:val_last_index]
    test_paths = image_paths[val_last_index:]

    copy_images(train_paths, Path(config.OUTPUT_PATH) / 'train')
    copy_images(val_paths, Path(config.OUTPUT_PATH) / 'val')
    copy_images(test_paths, Path(config.OUTPUT_PATH) / 'test')


