
'''
Dataset can be downloaded by:
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
'''

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt  
import os
from xml.etree import ElementTree

classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
               'sofa': 17, 'train': 18, 'tvmonitor': 19}  



def read(image_path, label_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h, image_w = image.shape[0:2]
    image = cv2.resize(image, (448, 448))
    image = image / 255.

    label_matrix = np.zeros([7, 7, 30])
    
    tree = ElementTree.parse(label_path)
    root = tree.getroot()
    boxes = list()
    for box in root.findall('.//object'):
        name = box.find('name').text   
        xmin = int(box.find('./bndbox/xmin').text)
        ymin = int(box.find('./bndbox/ymin').text)
        xmax = int(box.find('./bndbox/xmax').text)
        ymax = int(box.find('./bndbox/ymax').text)
        coors = [xmin, ymin, xmax, ymax, name]
        boxes.append(coors)
    
    label_matrix=np.zeros([7,7,30])
    for i in range(len(boxes)):
        box = boxes[i]
        ymin, ymax = box[1], box[3]
        xmin, xmax = box[0], box[2]
        name = classes_num[box[4]]
            
        x = (xmin + xmax) / 2 / image_w
        y = (ymin + ymax) / 2 / image_h
        w = (xmax - xmin) / image_w
        h = (ymax - ymin) / image_h
        loc = [7 * x, 7 * y]
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        y = loc[1] - loc_i
        x = loc[0] - loc_j

        if label_matrix[loc_i, loc_j, 24] == 0:
            label_matrix[loc_i, loc_j, name] = 1
            label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
            label_matrix[loc_i, loc_j, 24] = 1

    return image, label_matrix


