import torch
from torchvision import datasets, transforms
import numpy as np
def create_transformations():
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_test_transforms = [transforms.Resize(255), 
                             transforms.CenterCrop(224), 
                             transforms.ToTensor(), 
                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                  [0.229, 0.224, 0.225])]

    validation_transforms = transforms.Compose(valid_test_transforms)
    test_transforms = transforms.Compose(valid_test_transforms)
    
    return train_transforms, validation_transforms, test_transforms


def load_data(data_dir, batch_size=64):      
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"
   
    train_transforms, validation_transforms, test_transforms = create_transformations()
    
    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validationloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, validationloader, testloader, test_dataset.class_to_idx
    
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    new_width, new_height = 256, 256
    crop_width, crop_height = 224, 224
    left = (new_width - crop_width) / 2
    top = (new_height - crop_height) / 2
    right = (new_width + crop_width) /2
    bottom = (new_height + crop_height) / 2
    means, stds = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    
    image = image.resize((new_width, new_height))
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    np_image_scaled = np_image / 255 # convert the colour channel values from 0-255 to 0-1
    np_image_norm = (np_image_scaled - means) / stds
    processed_img = np_image_norm.transpose((2,0,1))
    return processed_img
    