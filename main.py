import torch
import torchvision
import torchvision.transforms as transforms
 
from efficientnet_pytorch import EfficientNet

import matplotlib
# print(matplotlib.use('Qt5Agg'))
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import random
import argparse

from module import *
from car_model import CarRecognizer

AVALIABLE_FORMATS = ['jpg','JPG','jpeg','JPEG','png','PNG']


# python main.py -k 2 --seed 'Ты не' --corpus narnija.txt --len 100


parser = argparse.ArgumentParser(description='Recognize car')
parser.add_argument("--source", help="Source image file or dir with files",required=True, type=str)
parser.add_argument("-e", help="Extended prediction view", action='store_true')
parser.add_argument("-n", help="Max number of source images from dir to recognize",default=-1, type=int)


args = parser.parse_args()
source = args.source
extended = args.e
n = args.n

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")



yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)
yolov5.eval()

ENClassifier = EfficientNet.from_name('efficientnet-b0',num_classes=5).to(device)
ENClassifier.load_state_dict(torch.load('ENClassifier.pth',map_location=device))
ENClassifier.eval()

model  = CarRecognizer(yolov5,ENClassifier,device)

matplotlib.use('TkAgg')

imgs = {}

if os.path.isdir(source):
    if not source.endswith('/'):
        source = source+"/"
    
    filenames = [source +name for name in os.listdir(source) if name[(name.rfind('.')+1):] in AVALIABLE_FORMATS]
    print(f"Found {len(filenames)} pics in {source}/")

    if len(filenames)==0:
        raise  FileNotFoundError(f'Dirrectory {source} contain no avaliable files! Only {AVALIABLE_FORMATS} formats are allowed') 

    for filename in filenames:
        im=Image.open(filename)
        imgs[filename] = im

elif os.path.isfile(source):
    if source[(source.rfind('.')+1):] in AVALIABLE_FORMATS:
        im=Image.open(source)
        imgs[source] = im
    else: raise NameError(f'Wrong format! Only {AVALIABLE_FORMATS} formats are allowed')
else:
    raise FileNotFoundError(f'Cant find file {source}')

# tfms = transforms.Compose([transforms.Resize((224, 224)),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize(0.5, 0.5, 0.5)])
# img = list(imgs.values())[0]

# # img = img / 2 + 0.5
# plt.imshow(img)
# plt.show()

recognize_images(imgs.values(),model,extended=extended,n=n)




