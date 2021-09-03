
import torch
import torchvision
import torchvision.transforms as transforms
 
import matplotlib.pyplot as plt
import numpy as np

from efficientnet_pytorch import EfficientNet

from PIL import Image,ImageStat

import json

from matplotlib.gridspec import GridSpec 
import math 
import os

import random



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

from module import *
from car_model import CarRecognizer


yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)
yolov5.eval()

ENClassifier = EfficientNet.from_name('efficientnet-b0',num_classes=5).to(device)
ENClassifier.load_state_dict(torch.load('ENClassifier.pth',map_location=device))
ENClassifier.eval()

model  = CarRecognizer(yolov5,ENClassifier,device)

img_folder_path = 'data'

import glob
imgs = []
for filename in glob.glob(f'{img_folder_path}/*.jpg'):
    im=Image.open(filename)
    imgs.append(im)
print(f"Found {len(imgs)} pics in {img_folder_path}/")

recognize_images(imgs,model,extended=False)




