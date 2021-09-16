# car-recognizer

This model recognizes vehicle of some classes.

At first yolov5 detect car in the picture and crop in out. Then EficientNet classifier decides which vehicle is in the picture

EficientNet classes: 'ambulance', 'common', 'fire-truck', 'police', 'tractor'



## demo recognization
![images](results/res.png)

1. ~~police (97.1%)~~
2. fire-truck (99.3%)
3. ~~[no car]~~
4. ~~[no car]~~
5. fire-truck (98.6%)
6. ~~ambulance (43.7%)~~
7. fire-truck (100.0%)
8. tractor (60.8%)
9. police (53.7%)
10. police (98.5%)
11. ~~[no car]~~
12. common (98.8%)
13. ambulance (95.6%)
14. common (60.0%)
15. ambulance (93.8%)
16. tractor (85.4%)
17. common (94.8%)
18. fire-truck (99.5%)
19. police (92.8%)
20. police (99.2%)

*not perfect, but not bad*

---
Development notebook : https://colab.research.google.com/drive/1RHvpvsnyW22qcUvnff35Zfj_F1gVnmJd?usp=sharing
