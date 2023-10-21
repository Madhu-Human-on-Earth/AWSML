import torch
from PIL import Image
import numpy as np

def process_image(im):
    std=[0.229, 0.224, 0.225]
    mean=[0.485, 0.456, 0.406];
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size=256,256
    #cropped_example = original.crop((left, top, right, bottom))
    
    im.thumbnail(size,Image.ANTIALIAS)
    im=im.crop((16,16,224+16,224+16))
    np_image = np.array(im)/255
#     print(np_image)
    for i in [0,1,2]:
        np_image[:,:,i]=(np_image[:,:,i]-mean[i])/std[i]
    final_image = np.transpose(np_image, (2,0,1))
    return final_image
def predict(image_path, model, topks, device,indexClass):
        image=process_image(Image.open(image_path))
        image=torch.FloatTensor([image])
        model.eval()
        output=model(image.to(device))
        prob=torch.exp(output.cpu())
        top_p,top_c = prob.topk(topks,dim=1)
#         print(type(idx_to_class))
        top_class = [indexClass.get(x) for x in top_c.numpy()[0]]
        return top_p,top_class
def ProcessType():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
