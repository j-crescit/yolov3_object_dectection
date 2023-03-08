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
with open(class_names_path, 'r') as f:
    class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
    f.close()

# load model

# load image

# convert image

# get detections

# bounding box, class id, confidence

# apply nms

# plot