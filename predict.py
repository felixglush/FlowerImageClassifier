import torch
import numpy as np
from model_utils import load_model
from get_input_args import get_predict_args
from utils import get_device
from PIL import Image
from image_preprocessor import process_image
import json

def main():
    args = get_predict_args()
    if args.topk < 1: 
        print("Exiting... topk arg must be greater than 0")
        exit()
    
    model = load_model(args.checkpoint)
    top_prob, top_class = predict(args.input, model, args.topk, model.class_to_idx, args.gpu)
    
    print("Top probabiliites:")
    print(*top_prob, sep=", ")
    
    print("Top classes:")
    print(*top_class, sep=", ")
    
    if args.category_names:
        # category_to_name is a mapping from the integer encoded categories to the text names of the flowers
        category_to_name = None
        with open(args.category_names, 'r') as f:
            category_to_name = json.load(f)
        class_names_list = convert_to_names(top_class, category_to_name)
       
        print("Top class names:")
        print(*class_names_list, sep=", ")

        
def convert_to_names(classes, cat_to_names_dict):
    return [cat_to_names_dict[str(cat)] for cat in classes]
    
    
def predict(image_path, model, topk, class_to_idx, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = get_device(gpu)
    print("Using device: ", device)

    image = Image.open(image_path)    
    processed_img = torch.from_numpy(process_image(image)).float().to(device)
    processed_img.unsqueeze_(0) # model expects batches, in this case a batch of 1 image
    model.to(device)
       
    with torch.no_grad():
        model.eval() 

        log_probs = model(processed_img)
        probs = torch.exp(log_probs)
        top_p, top_class_idx = probs.topk(topk)
        
        class_to_idx = model.class_to_idx
        
        idx_to_class = {str(val):int(key) for key, val in class_to_idx.items()} # invert class_to_idx
        top_class_idx = top_class_idx[0].cpu().numpy()
        top_class = np.array([idx_to_class[str(idx)] for idx in top_class_idx])
        
        top_probabilities = top_p[0].cpu().numpy()
        model.train()
        
        return top_probabilities, top_class 
    
    
if __name__ == "__main__":
    main()
