# Imports
import matplotlib.pyplot as plt
import argparse
import json

import torch
from torchvision import transforms
from PIL import Image


def parse():
    parser = argparse.ArgumentParser(description="Prediction paramenters")

    parser.add_argument('--model_path', help='Model path',
                        default='./checkpoint.pth', type=str)
    parser.add_argument(
        '--device', help='Choose GPU or CPU to handle the network', default='GPU', type=str)
    parser.add_argument('--image_path', help='Image path',
                        default='./flowers/train/1/image_06735.jpg', type=str)
    parser.add_argument('--category_names',
                        help='File to crossmatch category names', default='cat_to_name.json', type=str)
    parser.add_argument(
        '--topk', help='top classes returned', default=5, type=int)

    args = parser.parse_args()

    return args


def get_device(args):
    if args.device == 'GPU' and torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    return


def load_checkpoint(args):
    checkpoint = torch.load(args.model_path)
    model = checkpoint['base_model']

    for param in model.parameters():
        param.requieres_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def process_image(args):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(args.image_path)

    img_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return img_transform(img)


def get_category_names(args):
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def predict(args, model, category_names):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    top_labels = []

    with torch.no_grad():

        # process image to tensor
        img = process_image(args.image_path)

        img.to(args.device)
        model.to(args.device)

        input_img = img.unsqueeze(0)

        logps = model(input_img)
        ps = torch.exp(logps)
        topk = int(args.topk)
        top_p, top_class = ps.topk(topk, dim=1)

        # DEBUG
        class_idx = {val: key for key, val in model.class_to_idx.items()}

        for c in top_class.cpu().numpy().tolist()[0]:
            top_labels.append(class_idx[c])

        top_class_name = [category_names[str(idx)] for idx in top_labels]

    return top_p, top_labels, top_class_name


def main():
    args = parse()
    get_device(args)
    print("Loading model...")
    model = load_checkpoint(args)
    category_names = get_category_names(args)
    print("Making predictions...")
    top_p, top_labels, top_class_name = predict(args, model, category_names)
    print("Prediction results... \n")
    print(f'Top Prediction: {top_p}')
    print(f'Top Prediction label: {top_labels}')
    print(f'Top Prediction class name: {top_class_name}')


if __name__ == '__main__':
    main()
