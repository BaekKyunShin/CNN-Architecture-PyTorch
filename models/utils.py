from datetime import datetime 

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader

    Args:
        model (class): CNN model class
        data_loader (DataLoader): DataLoader for accuracy

    Returns:
        accuracy (float): total accuracy
    '''
    num_corrects = 0 
    n = 0
    with torch.no_grad():
        model.eval()
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            n += targets.size(0)
            num_corrects += (preds == targets).sum()
            accuracy = num_corrects.float() / n
    return accuracy


def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))
    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    # change the plot style to default
    plt.style.use('default')

def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image

def test(valid_loader, model, device):
    '''
    Function defining test
    
    Args:
        valid_loader (DataLoader): DataLoader for validating dataset
        model (class): CNN model class
        device: cuda or cpu
    '''
    dataiter = iter(valid_loader)
    images, labels = dataiter.next()
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    fig = plt.figure(figsize=(25, 4))

    for idx in np.arange(30):
        ax = fig.add_subplot(3, 10, idx+1, xticks=[], yticks=[])
        plt.imshow(im_convert(images[idx]))
        ax.set_title("{} ({})".format(str(CLASSES[preds[idx].item()]), str(CLASSES[labels[idx].item()])), color=("green" if preds[idx]==labels[idx] else "red"))
    plt.show()