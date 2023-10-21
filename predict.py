import argparse
import json
import torch
import numpy as np
from math import ceil
from PIL import Image
from train import check_gpu
from torchvision import models

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='Path to image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.')
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', dest="gpu", action="store_true", default=False)

    args = parser.parse_args()
    return args

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = models.vgg16(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image_path):
    img = Image.open(image_path)
    size = 256, 256
    img.thumbnail(size)
    width, height = img.size
    crop_size = 244
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    img = img.crop((left, top, right, bottom))
    np_image = np.array(img) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def predict(image_tensor, model, device, cat_to_name, topk=5):
    model.eval()
    with torch.no_grad():
        image_tensor = torch.from_numpy(image_tensor).float()
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        ps = torch.exp(output)
        top_probs, top_indices = ps.topk(topk, dim=1)
        top_probs = top_probs.cpu().numpy().tolist()[0]
        top_indices = top_indices.cpu().numpy().tolist()[0]
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_labels = [idx_to_class[idx] for idx in top_indices]
        top_flowers = [cat_to_name[label] for label in top_labels]
        return top_probs, top_labels, top_flowers

def print_probability(probs, flowers):
    for i, (flower, prob) in enumerate(zip(flowers, probs), 1):
        print(f"Rank {i}: Flower: {flower}, Likelihood: {ceil(prob * 100)}%")

def main():
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_checkpoint(args.checkpoint)
    
    image_tensor = process_image(args.image)
    
    device = check_gpu(args.gpu)
    
    top_probs, top_labels, top_flowers = predict(image_tensor, model, device, cat_to_name, args.top_k)
    
    print_probability(top_probs, top_flowers)

if __name__ == '__main__':
    main()
