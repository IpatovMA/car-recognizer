
import torch
import torchvision
import torchvision.transforms as transforms

from numpy.core.fromnumeric import size
from tqdm import tqdm
from PIL import Image
# import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import numpy as np
# Функции для нарезки картинки


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(
            1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0])              :int(xyxy[0, 2]), ::(1 if BGR else -1)]

    return crop


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

# Функция для классификатора


def predict(img, classifier, device=torch.device('cpu')):
    """
     Классифицирует картинку
    :param img: картинка 
    :param classifier: классификатор
    """
    classifier.eval()

    output_dict = {}
    with torch.no_grad():
        outputs = classifier(img.unsqueeze(0).to(device))

        for idx in torch.topk(outputs, len(outputs[0])).indices.squeeze(0).tolist():
            prob = torch.softmax(outputs, dim=1)[0, idx].item()
            output_dict[idx] = prob

    return output_dict


def display_prediction(img, prediction_dict, labels):
    """
    Отображает картинку и предсказания вероятности всех классов
    :param img: картинка 
    :param prediction_dict: словарь с предсказаниями
    :param labels: подписи классов
    """

    # Print predictions

    cell_text = []
    figsize=(20, 8)

    plt.subplots_adjust(right=0.5)
    for cls, prob in prediction_dict.items():
        cell_text.append([labels[cls], f'{prob*100:.2f}%'])
        print(f'{labels[cls]:<75} ({prob*100:.2f}%)')
    the_table = plt.table(cellText=cell_text, colLabels=[
                          'Class', "prob"], loc="right")
    imshow(img)
    

def imshow_in_ax(img,ax):
    """
    Отображает картинку в конкретных осях 
    :param img: нормализованный тензор - картинка
    :param ax: объект осей 
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_axis_off()


def imshow(img):
    """
    Отображает картинку 
    :param img: нормализованный тензор - картинка 
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    


def recognize_images(imgs, model, n=-1, extended=False):
    """
      Последовательно загружает картинки в модель
      :param imgs: список картинок
      :param model: модель
      # :param n: количество картинок из списка для распозанвания
      :param extended: отображать вероятности для всех классов у каждой картинки
      """
    pred_list = []
    classes = model.FINAL_CLASSES
    if n>0:
        imgs = list(imgs)[0:n]
    for i, img in tqdm(enumerate(imgs)):
        
        if extended:
            print(i+1)
            plt.subplot()
            model.recognize(img,display = True)
        else:
            pred_dict = model.recognize(img, display=False)
            if pred_dict != -1:
                idx = max(pred_dict, key=pred_dict.get)
                pred_list.append(
                    f'{i+1}.{classes[idx]} ({pred_dict[idx]*100:.1f}%)\n')
            else:
                pred_list.append(f'{i+1}.[no car]\n')

    if extended:
        return
    else:
        tfms = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(0.5, 0.5, 0.5)])
        tr_imgs = list(map(tfms, imgs))
        print(*pred_list)
        # fig = plt.figure(figsize=(20, 12), dpi=300)
        imshow(torchvision.utils.make_grid(tr_imgs))
        plt.savefig('results/res.png')
        with  open("results/res.txt", "w") as res:
            res.writelines(pred_list)
        
