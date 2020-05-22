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

COLORS=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

class PersonTracker:
    # set defaults
    img_size=416
    conf_thres=0.8
    nms_thres=0.4
    
    hashes={}
    tree=None

    # Tensor = torch.cuda.FloatTensor
    Tensor = torch.FloatTensor
    # The person count 
    PERSON_COUNT = 0

    def __init__(self, config_path, weights_path, class_path):
        # load model and put into eval mode
        self.model = Darknet(config_path, img_size=self.img_size)
        self.model.load_weights(weights_path)
        # model.cuda()
        self.model.eval()

        self.mot_tracker = Sort()

        self.classes = utils.load_classes(class_path)

    def detect_image(self, img):
        # scale and pad image
        ratio = min(self.img_size/img.size[0], self.img_size/img.size[1])
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
        input_img = Variable(image_tensor.type(self.Tensor))

        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = utils.non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)

        return detections[0]


    def index_person(self, image, distance = 15):
        # compute the hash for the image and convert it
        h = dhash(image)
        h = convert_hash(h)

        person_id = -1
        # search for similar person 
        if self.tree is not None:
            sim = sorted(tree.get_all_in_range(h, distance))
            if len(sim):
                p_ids = self.hashes.get(sim[0][1], [])
                if len(p_ids):
                    person_id = p_ids[0]

        if person_id == -1:
            PERSON_COUNT += 1
            person_id = PERSON_COUNT

        # update the hashes dictionary
        l = self.hashes.get(h, [])
        l.append(person_id)
        self.hashes[h] = l

        points = list(hashes.keys())
        self.tree = vptree.VPTree(points, hammingDistance)

        return person_id


    def process_detections(self, detections, frame):
        pad_x = max(frame.shape[0] - frame.shape[1], 0) * (self.img_size / max(frame.shape))
        pad_y = max(frame.shape[1] - frame.shape[0], 0) * (self.img_size / max(frame.shape))
        unpad_h = self.img_size - pad_y
        unpad_w = self.img_size - pad_x

        if detections is not None:
            tracked_objects = self.mot_tracker.update(detections.cpu())

            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * frame.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * frame.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * frame.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * frame.shape[1])
                color = COLORS[int(obj_id) % len(COLORS)]
                clss = self.classes[int(cls_pred)]

                if clss == "person":
                    cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 1)
                    # search for the person in the VP-Tree
                    # if 
                    cv2.putText(frame, str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
        
        return frame


    def process_video(self, videopath, sec=0):
        vid = cv2.VideoCapture(videopath)

        if not vid.isOpened():
            print("[ERROR] failed to read video")
            return 

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid.set(cv2.CAP_PROP_POS_MSEC, sec*1000)

        w = vid.get(cv2.CAP_PROP_WIDTH)
        h = vid.get(cv2.CAP_PROP_HEIGHT)

        frame_id = 0

        starttime = time.time()

        ext = videopath.split(".")[-1]

        outvideo = cv2.VideoWriter(videopath.replace(ext, "-det.mp4"), fourcc, 20.0, (w,h))

        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(frame)
            detections = self.detect_image(pilimg)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if detections is not None:
                frame = self.process_detections(detections, frame)

            outvideo.write(frame)

            frame_id += 1
            # if frame_id > 200:
            #     break

        vid.release()
        
        totaltime = time.time()-starttime
        print(frame_id, "frames", totaltime, "s")

if __name__ == "__main__":
    videopath = './videos/144p/the_laundromat.webm'
    config_path='config/yolov3.cfg'
    out_path='output/'
    weights_path='weights/yolov3.weights'
    class_path='config/coco.names'

    person_tracker = PersonTracker(config_path, weights_path, class_path)
    
    person_tracker.process_video(videopath, 406)