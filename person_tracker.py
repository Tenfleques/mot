#!/bin/python
import cv2
from sort import *
from hashing import convert_hash, hammingDistance, dhash
from imutils import paths
import argparse
import pickle
import vptree

from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image


# load weights and set defaults
config_path='config/yolov3.cfg'
out_path='output/'
weights_path='weights/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
# model.cuda()
model.eval()


classes = utils.load_classes(class_path)

# Tensor = torch.cuda.FloatTensor
Tensor = torch.FloatTensor
# The person count 
PERSON_COUNT = 0

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))

    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)

    return detections[0]


def trackPerson(image, hashes, tree, distance = 15):
    global PERSON_COUNT
    # compute the hash for the image and convert it
    h = dhash(image)
    h = convert_hash(h)

    person_id = -1
    # search for similar person 
    if not tree == None:
        sim = sorted(tree.get_all_in_range(h, distance))
        if len(sim):
            p_ids = hashes.get(sim[0][1], [])
            if len(p_ids):
                person_id = p_ids[0]

    if person_id == -1:
        PERSON_COUNT += 1
        person_id = PERSON_COUNT

    # update the hashes dictionary
    l = hashes.get(h, [])
    l.append(person_id)
    hashes[h] = l

    points = list(hashes.keys())
    tree = vptree.VPTree(points, hammingDistance)

    return tree, hashes, person_id

def processDetections(frame_id, detections, frame, hashes, tree):
    pad_x = max(frame.shape[0] - frame.shape[1], 0) * (img_size / max(frame.shape))
    pad_y = max(frame.shape[1] - frame.shape[0], 0) * (img_size / max(frame.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    person_id = 0
    p_id = None

    img = np.copy(frame)

    for x1, y1, x2, y2, obj_id, cls_pred, e in detections :
        cls = classes[int(cls_pred)]
        if cls == "person":
            box_h = int(((y2 - y1) / unpad_h) * frame.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * frame.shape[1])

            y1 = int(((y1 - pad_y // 2) / unpad_h) * frame.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * frame.shape[1])
            person = img[y1:y1+box_h, x1:x1+box_w, :]

            if person.shape[0] and person.shape[1]:
                # cv2.imwrite(filename, person)
                # breakdownPerson(person)
                tree, hashes, p_id = trackPerson(person, hashes, tree)
                person_id += 1

                # print(p_id)
        
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), (255,0,0), 1)
            
            if not p_id == None:
                cv2.putText(frame, str(p_id), (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)

    return frame, tree, hashes


videopath = './videos/144p/the_laundromat.webm'
videopath = './videos/144p/the_laundromat.webm'

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

frame_id = 0
sec = 406
fps = 12.0 # vid.get(cv2.CAP_PROP_FPS)

delta_time_between_frames = 1/fps

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vid.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
ret,frame = vid.read()

starttime = time.time()
tree = None 
hashes = {}
outvideo = cv2.VideoWriter(videopath.replace(".webm", "-det.mp4"),fourcc,20.0,(vw,vh))

if ret:
    vw = frame.shape[1]
    vh = frame.shape[0]

    vw = frame.shape[1]
    vh = frame.shape[0]

    while(ret):
        vid.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
        ret, frame = vid.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if detections is not None:
            frame, tree, hashes = processDetections(frame_id, detections, frame, hashes, tree)
        
        cv2.imshow('preview',frame)
        outvideo.write(frame)

        frame_id += 1
        sec += delta_time_between_frames

        # print(sec)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

totaltime = time.time()-starttime
print(frame_id, "frames", totaltime, "s")
cv2.destroyAllWindows()
outvideo.release()