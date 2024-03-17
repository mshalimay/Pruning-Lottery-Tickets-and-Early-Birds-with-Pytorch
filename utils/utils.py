from __future__ import print_function
import os
import csv
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import datasets, transforms
import torch.nn.init as init
import torch.nn as nn
import filters

# Training
def train(log_interval, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()        
        if log_interval > 0 and batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Training
def train_noprint(log_interval, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()        



def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.4f}%)\n')

    return test_loss, accuracy

def save_checkpoint(state, is_best, epoch, filepath, pr_iter):
    if is_best and epoch is not None:
        filename = os.path.join(filepath, f'priter={pr_iter}_epoch={epoch}_best.pth.tar')
    elif is_best and epoch is None:
        filename = os.path.join(filepath, f'priter={pr_iter}_best.pth.tar')
    else:
        filename = os.path.join(filepath, f'priter={pr_iter}_epoch={epoch}.pth.tar')
    torch.save(state, filename)
    print(f'Saved checkpoint to {filename}')

def log_training(log_path:str, data:dict) -> None:
    # Check if file is empty
    file_exists = os.path.isfile(log_path)
    file_is_empty = True if not file_exists else os.stat(log_path).st_size == 0
    
    # Create the file
    with open(log_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        # write the header if the file doesn't exist or is empty
        if not file_exists or file_is_empty:
            writer.writeheader()
        writer.writerow(data)

def dataset_num_classes(dataset_name:str):
    num_classes = 0
    if dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'mnist':
        num_classes = 10
    else:
        raise NotImplementedError
    return num_classes


def GPU_warmup(device, in_shape=(3,224,224), warmup_iterations=20):
    # Load the pre-trained model
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    # model = models.__dict__[model_name](pretrained=True)
    model.to(device)
    model.eval()

    dummy_data = torch.randn(1, in_shape[0], in_shape[1], in_shape[2], dtype=torch.float).to(device)

    # Perform warm-up iterations
    for _ in range(warmup_iterations):
        _ = model(dummy_data)

    del dummy_data
    del model

def load_preprocess_data(dataset:str, args) -> datasets:
    if dataset.lower() == 'mnist':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        data_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
        data_test = datasets.MNIST('./data', train=False, transform=transform)

    elif dataset.lower() == 'cifar10':
        transform_train=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.Lambda(lambda x: filters.my_gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
                           transforms.Lambda(lambda x: filters.my_gaussian_filter_2(x, 1/args.sigma, args.filter) if args.filter == 'highpass' else x),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(x)) if args.sparsity_gt > 0 else x),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])

        transform_test=transforms.Compose([
                           transforms.Lambda(lambda x: filters.my_gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
                           transforms.Lambda(lambda x: filters.my_gaussian_filter_2(x, 1/args.sigma, args.filter) if args.filter == 'highpass' else x),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(x)) if args.sparsity_gt > 0 else x),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])
        data_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        data_test = datasets.CIFAR10('./data', train=False, transform=transform_test)
    else:
        raise NotImplementedError

    return data_train, data_test



# Function for Initialization
def weight_init(model):
    if isinstance(model, nn.Conv1d):
        init.normal_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.Conv2d):
        init.xavier_normal_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.Conv3d):
        init.xavier_normal_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.ConvTranspose1d):
        init.normal_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.ConvTranspose2d):
        init.xavier_normal_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.ConvTranspose3d):
        init.xavier_normal_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.BatchNorm1d):
        init.normal_(model.weight.data, mean=1, std=0.02)
        init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm2d):
        init.normal_(model.weight.data, mean=1, std=0.02)
        init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm3d):
        init.normal_(model.weight.data, mean=1, std=0.02)
        init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.Linear):
        init.xavier_normal_(model.weight.data)
        init.normal_(model.bias.data)
    elif isinstance(model, nn.LSTM):
        for param in model.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(model, nn.LSTMCell):
        for param in model.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(model, nn.GRU):
        for param in model.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(model, nn.GRUCell):
        for param in model.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
