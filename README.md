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
3. Switch to venv: `source venv/bin/activate`
3. Install required packages: `pip install -r requirements.txt`


## Interference

Script `scripts/predict.py` is responsible for interference.
Please type in following command in command line to perform interference on some photo.

`python scripts/predict.py -i <path_to_image>`

Some example photos are located in `data/interference` folder.

Image with confidence score and predicted flower will pop up. Click `Enter` to close window.

## Training 

Model was trained on flowers dataset taken from: [kaggle.com/datasets/alxmamaev/flowers-recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
I decided to fine-tune ResNet34 model with new FCN at the end of the network. I decided to train two models for 20 epochs, saving version from epoch with the best performance on validation loss. Firstly I decided to train network with all parameters unfrozen. This resulted in a quite unstable learning (probably it would be better, after longer time). Later, i decided to train model, with convolutional part frozen. Taking into acount that models were trained only for 20 epochs, the latter approach turned out to be better, with higher accuracy and more stable learning. I saved the models, from the epoch with the best validation loss. Models parameters and plots with learning history are held in `models` folder
In order to same training please follow these steps:

1. Download flowers dataset from [link](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition), unzip it and 
place `flowers` folder  in `classfication_flowers/data` directory
2. From root folder run `python scripts/build_dataset.py`. This splits data into format accessible for model training
3. `configs/train_config.py` contains all training parameters, as well as model saving path. If you want you can change it.
4. Run `scripts/train.py` from root folder.


## Additional tasks
I performed following additional tasks:
### Write unit tests for the 3 most important classes/functions
  Tests for three functions can be found in `tests/unit_tests` folder.
### Use FastAPI to handle the model
This is done by `fast.api.py`. For interference please type in following command in root folder
`uvicorn fast_api:app --reload`

By default this service is run on 8000 port. To get more details about arguments pass:  
`http://127.0.0.1:8000/docs`  
To infer from model, use GET with parameters listed in docs, and image_path indicating location of image on your machine.  
E.g `http://127.0.0.1:8000/infer/device=cpu&model_name=backbone_frozen&image_path=/home/skocznapanda/programming/classification_deeptale/data/interference/example_1.jpg`    
The response gives you predicted class and model confidence    
E.g `[["tulip",0.9980118274688721]]`

### Use pre-commit
I installed pre-commit and added `.pre-commit-config.yaml`  to the directory root.


