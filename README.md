# Car recognize system

This model recognizes vehicle of some classes.

At first yolov5 detect car in the picture and crop it out. Then EficientNet classifier decides which vehicle is in the picture

EficientNet classes: `ambulance`, `common`, `fire-truck`, `police`, `tractor`


```
usage: main.py [-h] --source SOURCE [-e] [-n N]

Recognize car

optional arguments:
  -h, --help       show this help message and exit
  --source SOURCE  Source image file or dir with files
  -e               Extended prediction view
  -n N             Max number of source images from dir to recognize
```

## Demo recognization

```
python main.py --source data -n 16

Using torch 1.9.0+cpu (CPU)
Using cache found in /home/mikhail/.cache/torch/hub/ultralytics_yolov5_master
YOLOv5 🚀 2021-9-3 torch 1.9.0+cpu CPU

Fusing layers... 
[W NNPACK.cpp:79] Could not initialize NNPACK! Reason: Unsupported hardware.
Model Summary: 224 layers, 7266973 parameters, 0 gradients
Adding AutoShape... 
Found 20 pics in data//
0it [00:00, ?it/s]Cant find vehicle in the picture
8it [00:07,  1.01s/it]Cant find vehicle in the picture
16it [00:14,  1.08it/s]

 1.[no car]
 2.fire-truck (99.3%)
 3.fire-truck (98.6%)
 4.ambulance (43.7%)
 5.fire-truck (100.0%)
 6.tractor (60.8%)
 7.police (53.7%)
 8.police (98.5%)
 9.[no car]
 10.common (98.8%)
 11.police (97.1%)
 12.ambulance (95.6%)
 13.common (60.0%)
 14.ambulance (93.8%)
 15.tractor (85.4%)
 16.common (94.8%)

```

![images](results/res.png)


---
[Development notebook](./yolov5_car.ipynb) [<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/mipatov/car-recognizer/blob/master/yolov5_car.ipynb)

