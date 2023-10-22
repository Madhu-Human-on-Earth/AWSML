import torch
from PIL import Image
from torchvision import transforms

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

def predict(image_path, model, topk, device, class_to_idx):
    model.eval()
    image_tensor = process_image(image_path)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model.forward(image_tensor)
        probabilities, indices = torch.topk(torch.exp(output), topk)
        probabilities = probabilities.squeeze().tolist()
        indices = indices.squeeze().tolist()
    
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    top_classes = [idx_to_class[index] for index in indices]
    
    return probabilities, top_classes

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

def main():
    checkpoint_path = 'checkpoint.pth'
    image_path = 'image.jpg'
    topk = 5
    
    class_to_idx = {
        'class1': 0,
        'class2': 1,
        # Add more class to index mappings here
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    
    probabilities, top_classes = predict(image_path, model, topk, device, class_to_idx)
    
    print(f"Top {topk} classes:")
    for probability, class_idx in zip(probabilities, top_classes):
        print(f"Class Index: {class_idx}, Probability: {probability:.4f}")

if __name__ == '__main__':
    main()
