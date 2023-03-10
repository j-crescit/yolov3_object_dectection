import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import util

# define constants
cfg_path = os.path.join('.', 'model', 'cfg', 'yolov3.cfg')
weights_path = os.path.join('.', 'model', 'weights', 'yolov3.weights')
names_path = os.path.join('.', 'model', 'class.names')

# load class names
with open(names_path, 'r') as f:
    class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
    f.close()

# load model
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# load image
img = cv2.imread('image/dog.jpg')

# convert image
blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), True)

# get detections
net.setInput(blob)

# bounding box, class id, confidence

# apply nms

# plot