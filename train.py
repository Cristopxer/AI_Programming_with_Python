# Imports
from workspace_utils import active_session

import argparse
import json

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


def parse():

    parser = argparse.ArgumentParser(description="Network parameters")

    parser.add_argument(
        '--data_dir', help='Directory where the data is stored', default='./flowers', type=str)
    parser.add_argument(
        '--save_dir', help='Directory where the data is saved', default='./', type=str)
    parser.add_argument(
        '--model', help='Network architecture vgg16/densnet121', default='vgg16', type=str)
    parser.add_argument(
        '--device', help='Choose GPU or CPU to handle the network', default='GPU', type=str)
    parser.add_argument('--batch_size', help='Batch size to process images',
                        default=32, type=int)
    parser.add_argument('--lr', help='Learning rate',
                        default=0.001, type=float)
    parser.add_argument('--epochs', help='Number of epochs',
                        default=3, type=int)
    parser.add_argument('--dropout', help='Dropout rate',
                        default=0.2, type=float)
    parser.add_argument('--out_classes', help='Number of output classes',
                        default=102, type=int)

    return parser.parse_args()


def get_device(args):
    if args.device == 'GPU' and torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    return


# LOAD DATA FUNCTIONS

def set_dir(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    return [train_dir, valid_dir, test_dir]


def transform_data(data_dir, batch_size):
    train_dir, valid_dir, test_dir = data_dir

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    data_sets = {'train_data': train_data,
                 'valid_data': valid_data,
                 'test_data': test_data}
    loaders = {'train_loader': train_loader,
               'valid_loader': valid_loader,
               'test_loader': test_loader}

    return data_sets, loaders


def get_class_labels():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def build_model(args):

    if args.model == 'vgg1':
        model = models.vgg16(pretrained=True)
        input_units = 25088
        hidden_units = [4096, 1024]

    elif args.model == 'densnet121':
        model = models.densenet121(pretrained=True)
        input_units = 1024
        hidden_units = [512, 256]

    for param in model.parameters():
        param.requires_grad = False

    dropout = args.dropout
    out_classes_num = args.out_classes
    model.classifier = nn.Sequential(nn.Linear(input_units, hidden_units[0]),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(
                                         hidden_units[0], hidden_units[1]),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(
                                         hidden_units[1], out_classes_num),
                                     nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()

    # optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    # Set model to the available device 'GPU' or 'CPU'

    model.to(args.device)

    return model, optimizer, criterion


def train(args, loaders, model, optimizer, criterion):

    epochs = args.epochs
    steps = 0
    print_every = 10

    for epoch in range(epochs):

        running_loss = 0
        for images, labels in loaders['train_loader']:
            steps += 1

            # move input and labels to default tensor device "cuda" or "cpu"
            images, labels = images.to(args.device), labels.to(args.device)

            # set optimizer, losgps and loss
            optimizer.zero_grad()

            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in loaders['valid_loader']:

                        # move test input and values to default tensor device
                        images, labels = images.to(
                            args.device), labels.to(args.device)
                        # test prediction
                        logps = model(images)

                        # calculate batch loss
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        # calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                valid_dataset_length = len(loaders['valid_loader'])

                print(f'Epoch {epoch + 1}/{epochs} - '
                      f'Train loss: {running_loss/print_every:.3f} -'
                      f'Test loss: {test_loss/valid_dataset_length:.3f} - '
                      f'Test accuracy: {accuracy/valid_dataset_length:.3f}')

                running_loss = 0
                model.train()

    return model


def model_validator(args, model, loader):
    model.eval()
    accuracy = 0

    for images, labels in loader['test_loader']:

        images, labels = images.to(args.device), labels.to(args.device)

        logps = model(images)

        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

    return accuracy/len(loader['test_loader'])


def save_checkpoint(model, args, optimizer, data_sets):
    model.class_to_idx = data_sets['train_data'].class_to_idx

    save_dir = args.save_dir + 'checkpoint.pth'

    torch.save({
        'epoch': args.epochs,
        'batch_size': args.batch_size,
        'base_model': models.vgg16() if args.model == 'vgg16' else models.densenet121(),
        'classifier': model.classifier(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': args.lr,
        'class_idx': model.class_to_idx,
    }, save_dir)

    print(f'model successfully saved in {save_dir}')


def main():
    args = parse()
    get_device(args)
    print(args.device)
    data_dir = set_dir(args)
    data_sets, loaders = transform_data(data_dir, args.batch_size)
    model, optimizer, criterion = build_model(args)
    with active_session():
        model = train(args, loaders, model, optimizer, criterion)
        model_accuracy = model_validator(args, model, loaders)
        print(f'--> model_accuracy: {model_accuracy} <--')
        save_checkpoint(model, args, optimizer, data_sets)


if __name__ == '__main__':
    main()
