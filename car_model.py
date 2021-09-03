import module
from module import *
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image,ImageStat

class CarRecognizer():
  def __init__(self,yolo,classifier,device = torch.device('cpu')):
    self.yolo = yolo
    self.classifier = classifier
    self.YOLO_CLASSES = [ 'car', 'truck', 'bus']
    self.FINAL_CLASSES = ['ambulance', 'common', 'fire-truck', 'police', 'tractor']
    self.device = device

  def recognize(self,img,display = True):
    """
    Распознает машину на картинке
    :param img: PIL картинка
    """
    # распознаем картинку с помощью yolo
    yolo_res = self.yolo(img)

    # отрезаем лишнее, оставяем наибольший кусок
    isvehicle = False # флаг машины на картинке

    names = yolo_res.names
    _, det = next(enumerate(yolo_res.pred))
    main_crop = img.copy()
    max_size = (1,1)
    
    for *xyxy, conf, cls in reversed(det):
      c = int(cls)  # integer class
      if names[c] in self.YOLO_CLASSES: 
          isvehicle = True
          crop= save_one_box(xyxy, np.array(img), BGR=True)
          crop = Image.fromarray(crop)
          if crop.size>max_size:
              max_size = crop.size
              main_crop = crop
    
    if not isvehicle:
      print("There is no vehicle in the picture")
      return -1
    
    # Нормируем картинку
    tfms = transforms.Compose([transforms.Resize((224,224)),
                           transforms.ToTensor(),
                           transforms.Normalize(0.5, 0.5, 0.5)])
    main_crop = tfms(main_crop)

    pred = predict(main_crop,self.classifier,self.device)  
    if display:
      display_prediction(main_crop,pred,self.FINAL_CLASSES) 

    return pred


