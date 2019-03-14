import torch
from torch import nn
from image_preprocessor import load_data
from model_utils import initialize_model, create_optimizer, save_model, load_model
from get_input_args import get_train_args
import constants
from utils import get_device
from workspace_utils import active_session
import os

def main():
    args = get_train_args()
    
    trainloader, validloader, testloader, class_to_idx = load_data(data_dir = args.data_dir)
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    train(save_dir = args.save_dir, 
          arch = args.arch, lr = args.learning_rate, 
          epochs = args.epochs, hidden_units = args.hidden_units, 
          gpu = args.gpu, continue_training = args.continue_training, 
          save_threshold = args.save_threshold, optimizer_name = args.optimizer_name,
          class_to_idx = class_to_idx, trainloader = trainloader, 
          validationloader = validloader)


def train(save_dir, arch, lr, epochs, hidden_units, gpu, 
          continue_training, trainloader, validationloader,
          save_threshold, optimizer_name, class_to_idx):
    print("LR", lr)
    print("Epochs", epochs)
    print("Hidden units", hidden_units)
    device = get_device(gpu)
    print("Using device: ", device)
    
    model = initialize_model(arch, constants.num_classes, hidden_units, constants.dropout)
    model.to(device)
    optimizer = create_optimizer(optimizer_name, model.classifier.parameters(), lr)
    criterion = nn.NLLLoss()
    
    steps = 0
    best_accuracy = 0
    train_loss = 0
    train_losses, valid_losses = [], []

    if continue_training:
        pass
    
    with active_session():
        print("Starting training")
        for e in range(epochs):
            print(f"Training epoch # {e + 1}")
            for images, labels in trainloader:
                steps += 1
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                if steps % 5 == 0:
                    valid_loss = 0 
                    accuracy = 0
                    model.eval()

                    with torch.no_grad():
                        for images, labels in validationloader:
                            images, labels = images.to(device), labels.to(device)

                            log_probs = model(images)
                            probs = torch.exp(log_probs)

                            loss = criterion(log_probs, labels)
                            valid_loss += loss.item()

                            top_p, top_class = probs.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor))
                    model.train()
                    acc_percentage = accuracy / len(validationloader)
                    train_losses.append(train_loss / len(trainloader))
                    valid_losses.append(valid_loss / len(validationloader))
                    print("Epoch: {}/{}.. ".format(e + 1, epochs),
                          "Training Loss: {:.3f}.. ".format(train_loss / len(trainloader)),  
                          "Validation Loss: {:.3f}.. ".format(valid_loss / len(validationloader)), 
                          "Validation Accuracy: {:.3f}".format(acc_percentage))
                    train_loss = 0

                    if acc_percentage > best_accuracy:
                        best_accuracy = acc_percentage
                        print("Best accuracy so far: {:.3}".format(best_accuracy.item()))

                        checkpoint_name = arch + "_flower_classifier2.pth"
                        save_model(model_state_dict = model.state_dict(), optim_state_dict = optimizer.state_dict(), 
                                   accuracy = acc_percentage, epoch = e, hidden_units = hidden_units, train_loss = train_loss,
                                   learningrate = lr, class_to_idx = class_to_idx, save_dir = save_dir, 
                                   checkpoint_name = checkpoint_name, model_name = arch)                                          
        print("Done training")

    
if __name__ == "__main__":
    main()