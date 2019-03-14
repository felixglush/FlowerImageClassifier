import torch
from torch import save, load
from torchvision import models
from torch import nn, optim
import constants
from collections import OrderedDict
 

def load_model(path, continue_training=False):
    print("Loading checkpoint from", path)
    checkpoint = torch.load(path)
    
    model_name = checkpoint["model_name"]
    model = initialize_model(model_name, constants.num_classes, checkpoint["hidden_size"], constants.dropout)
    optimizer = None
    
    accuracy = checkpoint["validation_accuracy"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    print("Loaded a model with accuracy", accuracy.item())
    
    if continue_training:
        optimizer = create_optimizer(constants.default_optimizer, model.classifier.parameters(), checkpoint["learningrate"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        epoch = checkpoint["epoch"]
        train_loss = checkpoint["train_loss"]
    
    return model

def save_model(model_state_dict, optim_state_dict, accuracy, 
               epoch, learningrate, hidden_units, train_loss, class_to_idx, 
               save_dir, checkpoint_name, model_name):
    checkpoint = {
        "model_name": model_name,
        "model_state_dict": model_state_dict,
        "optim_state_dict": optim_state_dict,
        "validation_accuracy": accuracy,
        "epoch": epoch,
        "learningrate": learningrate,
        "hidden_size": hidden_units,
        "train_loss": train_loss,
        "class_to_idx": class_to_idx,
    }
    torch.save(checkpoint, save_dir + "/" + checkpoint_name)
    print("Model saved")
    

def initialize_model(arch, num_classes, hidden_units, dropout):
    model = None
    
    if arch == "vgg16" or arch == "vgg11":
        if arch == "vgg16":
            model = models.vgg16(pretrained = True)
        else:
            model = models.vgg11(pretrained = True)    
        freeze_model_features(model)
        my_classifier = create_model_classifier(model, model.classifier[0].in_features, num_classes, hidden_units, dropout)
        model.classifier = my_classifier
    else:
        print("Invalid model name, exiting...")
        exit()
    return model
   
    
def create_model_classifier(model, input_units, num_classes, hidden_units, dropout):
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_units, hidden_units)),
        ('relu1', nn.ReLU(inplace=True)),
        ('dropout1', nn.Dropout(p=dropout)),

        ('fc2', nn.Linear(hidden_units, hidden_units)),
        ('relu2', nn.ReLU(inplace=True)),
        ('dropout2', nn.Dropout(p=dropout)),

        ('fc3', nn.Linear(hidden_units, num_classes)),
        ('output', nn.LogSoftmax(dim=1)),
    ]))
    
    
def freeze_model_features(model):
    for param in model.features.parameters():
        param.requires_grad = False
      
    
def create_optimizer(name, trainable_params, lr):
    if name == "adam":
        return optim.Adam(trainable_params, lr = lr)
    else:
        print("Exiting... Not a valid optimizer ", name)
        exit()
      