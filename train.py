import argparse
import torch
from collections import OrderedDict
from torch import nn, optim
from torchvision import models, transforms, datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for flower classification.")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, type=float)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store_true", default=False)
    return parser.parse_args()

def create_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def load_data(data_dir):
    train_transforms = create_transforms(train=True)
    valid_transforms = create_transforms(train=False)
    
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=valid_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, validloader, testloader, train_data.class_to_idx

def create_model(arch, hidden_units):
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.3)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    return model

def train_model(model, trainloader, validloader, criterion, optimizer, epochs, device):
    model.to(device)
    steps = 0
    print_every = 20
    
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validate_model(model, validloader, criterion, device)
                
                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                running_loss = 0
                model.train()

def validate_model(model, validloader, criterion, device):
    valid_loss = 0
    accuracy = 0
    for inputs, labels in validloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def save_checkpoint(model, save_dir, arch, hidden_units, class_to_idx):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, save_dir)
    print("Model saved successfully!")

def main():
    args = parse_args()
    trainloader, validloader, testloader, class_to_idx = load_data('flowers')
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    model = create_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    train_model(model, trainloader, validloader, criterion, optimizer, args.epochs, device)
    print("\nTraining process is completed!")
    
    model.eval()
    accuracy = validate_model(model, testloader, criterion, device)[1]
    print('Accuracy on test images: {:.2f}%'.format(accuracy*100))
    
    save_checkpoint(model, args.save_dir, args.arch, args.hidden_units, class_to_idx)

if __name__ == '__main__':
    main()
