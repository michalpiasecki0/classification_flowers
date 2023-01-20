## Flower image classificator



This repository contains model, which classify flowers into one of 5 categories:
1. daisy
2. dandelion
3. rose
4. sunflower
5. tulip


## Installation

Please follow these steps:

1. Move to directory where you want to save this project
1. Clone directory: `git clone git@github.com:michalpiasecki0/classification_flowers.git`
2. Create venv for this project: `python3 -m venv venv`
3. Install required packages: `pip install -r requirements.txt`


## Interference

Script `scripts\predict.py` is responsible for interference.
Please type in following command in command line to perform interference on some photo.

`python scripts/predict.py -i <path_to_image>`

Some example photos are located in `data/interference` folder.

Image with confidence score and predicted flower will pop up. Click `Enter` to close window.

## Training 

Model was trained on flowers dataset taken from: [link](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
In order to same training please follow these steps:

1. Download flowers dataset from [link](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition), unzip it and 
place `flowers` folder  in `classfication_flowers/data` directory
2. From root folder run `python scripts/build_dataset.py`. This splits data into format accessible for model training
3. `configs/train_config.py` contains all training parameters, as well as model saving path. If you want you can change it.
4. Run `scripts/train.py` from root folder.

