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
H, W, _ = img.shape

# convert image
blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), True)

# get detections
net.setInput(blob)
detections = util.get_outputs(net)

# bounding box, class id, confidence
b_boxes = []
c_id = []
confidence = []

for detection in detections:
    b_box = detection[:4]
    b_box = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

    b_box_confidence = detection[4]

    class_id = np.argmax(detection[4])
    score = np.amax(detection[5:])

    b_boxes.append(b_box)
    c_id.append(class_id)
    confidence.append(score)

# apply nms
b_boxes, c_id, confidence = util.NMS(b_boxes, class_id, confidence)

# plot