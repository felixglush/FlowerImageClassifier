import argparse
import constants

def get_train_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the training program from a terminal window. This function uses Python's 
    argparse module to created and defined these command line arguments.
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object  
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_dir", type = str, help = "path to images i.e. 'flowers'")
    parser.add_argument("--save_dir", type = str, default="checkpoints", help = "path to save model in i.e. 'checkpoints'")
    parser.add_argument("--arch", type = str, default="vgg16", choices = constants.supported_arch_list, help = "CNN model architecture")
    parser.add_argument("--learning_rate", type = float, default="0.001", help = "learning rate")
    parser.add_argument("--epochs", type = int, default="15", help = "number of epochs")
    parser.add_argument("--hidden_units", type = int, default="4096", help = "number of hidden layer units")
    parser.add_argument("--gpu", action="store_true", dest="gpu", help = "train on GPU if one is available")
    parser.add_argument("--continue_training", type = str, help = "path to a model to continue training it")
    parser.add_argument("--save_threshold", type = float, default=0.75, help = "0.xx accuracy threshold to save model at")
    parser.add_argument("--optimizer_name", type = str, default="adam", help = "optimizer to use for training model")
   
    return parser.parse_args()


def get_predict_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the prediction program from a terminal window. This function uses Python's 
    argparse module to created and defined these command line arguments.
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object  
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("input", type = str, help = "path to image")
    parser.add_argument("checkpoint", type = str, help = "path to model checkpoint")
    parser.add_argument("--topk", type = int, default=3, help = "top K predicted classes")
    parser.add_argument("--category_names", type = str, default="cat_to_name.json", help = "filename of mapping from classes to names")
    parser.add_argument("--gpu", action="store_true", dest="gpu", help = "perform inference on GPU if one is available")

    return parser.parse_args()
