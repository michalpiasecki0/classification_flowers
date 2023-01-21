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

Script `src/predict.py` is responsible for interference.
Please type in following command in command line to perform interference on some photo.

`python src/predict.py -i <path_to_image>`  

Image with confidence score and predicted flower will pop up. Click `Enter` to close window.

E.g  
`python src/predict.py -i data/interference/example_1.jpg`

Some example photos are located in `data/interference` folder.


## Training 

Model was trained on flowers dataset taken from: [kaggle.com/datasets/alxmamaev/flowers-recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
I decided to fine-tune ResNet34 model with new FCN at the end of the network. I decided to train two models for 20 epochs, saving version from epoch with the best performance on validation loss. Firstly I decided to train network with all parameters unfrozen. This resulted in a quite unstable learning (probably it would be better, after longer time). Later, i decided to train model, with convolutional part frozen. Taking into acount that models were trained only for 20 epochs, the latter approach turned out to be better, with higher accuracy and more stable learning. I saved the models, from the epoch with the best validation loss. Models parameters and plots with learning history are held in `models` folder
In order to same training please follow these steps:

1. Download flowers dataset from [link](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition), unzip it and 
place `flowers` folder  in `classfication_flowers/data` directory
2. From root folder run `python src/build_dataset.py`. This splits data into format accessible for model training
3. `conf/train_config.py` contains all training parameters, as well as model saving path. If you want you can change it.
4. Run `src/train.py` from root folder.


## Additional tasks
I performed following additional tasks:
### Write unit tests for the 3 most important classes/functions
  Tests for three functions can be found in `tests/unit_tests` folder.
### Use FastAPI to handle the model
This part is implemented in `main.py`.  
To start a server, please type in in command line:  
`uvicorn main:app --reload`

User can do following actions:
1. List all images ready for interference.
2. Upload new image for interference.
3. Delete image from images stored for interference.
4. Get prediction on one of images stored in folder for interference. User choose image, specify model and device.

Example command for interference:  
`http://127.0.0.1:8000/infer/?image_name=rose.jpg&device=cpu&model_name=backbone_frozen`  
Returns class and model confidence:  
`[
  [
    "rose",
    0.9762430787086487
  ]
]
`  
For more details please go to `http://127.0.0.1:8000/docs`, where everything is nicely described.


### Use pre-commit
I installed pre-commit and added `.pre-commit-config.yaml`  to the directory root.
`pre-commit-config.yaml` contains black and mypy. Torch libraries (E.g torchvision) do not
have type-hints, so as a workaround mypy ignores missing import error.


