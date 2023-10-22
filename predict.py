import argparse
import json
import torch
from PIL import Image
import numpy as np
from torch import nn
from torchvision import models

def parse_args():
    parser = argparse.ArgumentParser(description="Predict a flower name from an image.")
    parser.add_argument('--image', type=str, help='Path to image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.')
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', dest="gpu", action="store_true", default=False)
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    class_to_idx = checkpoint['class_to_idx']
    
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.3)),
        ('fc2', nn.Linear(hidden_units, len(class_to_idx))),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = class_to_idx
    
    return model

def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image)
    return image_tensor

def predict(image_path, model, topk, device):
    model.eval()
    image_tensor = process_image(image_path)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model.forward(image_tensor)
        probabilities, indices = torch.topk(torch.exp(output), topk)
        probabilities = probabilities.squeeze().tolist()
        indices = indices.squeeze().tolist()
    
    return probabilities, indices

def load_category_names(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def main():
    args = parse_args()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()
    
    image_path = args.image
    topk = args.top_k if args.top_k else 1
    cat_to_name = load_category_names(args.category_names)
    
    probabilities, indices = predict(image_path, model, topk, device)
    flower_names = [cat_to_name[str(index)] for index in indices]
    
    print(f"Top {topk} flower prediction:")
    for flower, probability in zip(flower_names, probabilities):
        print(f"Flower: {flower}, Probability: {probability:.4f}")

if __name__ == '__main__':
    main()
