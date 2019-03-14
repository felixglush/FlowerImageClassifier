# FlowerImageClassifier

This is the final project for the Udacity AI Programming with Python nanodegree. It consists of a CLI app that allows you to train a neural network and output predictions given input images. It leverages feature extraction to classify 102 different types of flowers.

It achieved 88.1% accuracy on the test dataset.

## Usage
1. Jupyter Notebook
2. Command Line Interface

## Commands
### train.py
Choose a data directory

`python train.py data_directory`

Choose where to save the checkpoint models

`python train.py data_dir --save_dir save_directory`

Choose a model architecture (currently supported: vgg16 and vgg11)

`python train.py data_dir --arch "vgg16"`

Set hyperparameters (defaults: learning rate = 0.001, hidden_units = 4096, epochs = 15)

`python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`

Use GPU for training

`python train.py data_dir --gpu`

### predict.py
Basic usage

`python predict.py /path/to/image checkpoint`

Return top K most likely classes

`python predict.py input checkpoint --top_k 3`

Use a mapping of categories to real names

`python predict.py input checkpoint --category_names cat_to_name.json`

Use GPU for inference

`python predict.py input checkpoint --gpu`
