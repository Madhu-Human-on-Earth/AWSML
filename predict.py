import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from PIL import Image
import glob, os
import utilize
import model_functions as fm
import argparse

def predicts(args):
    import json
    print("loading the checkpoint")
    model=fm.loadCheckpoint(args.checkpoint_path)
    print("Loading is finished")

    device="cpu"
    if(args.gpu==True):
        device=utilize.ProcessType()
    model.to(device);
    indexClass={}
    for i,value in model.class_to_idx.items():
        indexClass[value]=i
    probs, classes = utilize.predict(args.input, model,args.top_k,device,indexClass)
    print("Most "+str(args.top_k)+" Classes with their probabilities" )
    if(args.category_names):
        flower_to_name=[]
        with open(args.category_names, 'r') as f:
            flower_to_name = json.load(f)
        classes_label=[]
        print(probs.shape)
        for i in classes:
            classes_label.append(flower_to_name.get(str(i)))
        for i in range(args.top_k):
            print ("Class name: ",classes_label[i],",Probability :" ,probs[0][i].detach().numpy()*100)
    else:
        for i in range(args.top_k):
            print("Class #", classes[i],"Probability: ",probs[0][i].detach().numpy()*100)


def main():
    parser = argparse.ArgumentParser(description='Flowers Classifcation Predictor')
    parser.add_argument('--input', type=str, help='path for image to predict',required=True)
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='path to en existance  checkpoint',required=True)
    parser.add_argument('--top_k', type=int, default=5, help='top k classes for the input')
    parser.add_argument('--category_names', type=str, help='json path file of categories names of flowers')
    parser.add_argument('--gpu' , type=bool, default=False, help='checkpoint directory path')
    args = parser.parse_args()
    print(args)
    
    predicts(args)
    print("\nPrediction is finished\n")
if __name__ == "__main__":
    main()
