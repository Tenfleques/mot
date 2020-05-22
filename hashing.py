# import the necessary packages
import numpy as np
import cv2
import time
import itertools
import os
import dlib

import numpy as np
np.set_printoptions(precision=2)

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
dlibFacePredictor =os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
align = openface.AlignDlib(dlibFacePredictor)
imgDim = 96
net = openface.TorchNeuralNet(networkModel, imgDim)

def dhash(image, hashSize=8):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(gray, (hashSize + 1, hashSize))

    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]

    print(diff.flatten())
    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def convert_hash(h):
    # convert the hash to NumPy's 64-bit float and then back to
    # Python's built in int
    return int(np.array(h, dtype="float64"))

def hammingDistance(a, b):
    # compute and return the hamming distance between the integers
    return bin(int(a) ^ int(b)).count("1")

def euclidean(p1, p2):
    # compute and return the euclidean distance between the integers
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))


def feature_distance(a,b):
    d = a["data"] - b["data"]
    return np.dot(d, d)


def featurehash(rgbImg, hashSize=96):
    # convert the image to rgb
    # rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbImg)
    
    if bb is None:
        print("Unable to find a face:")
        cv2.imshow("image", rgbImg)
        key = cv2.waitKey(1)
        return None
    
    alignedFace = align.align(hashSize, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    if alignedFace is None:
        print("Unable to align image:")
        rgbImg = cv2.resize(rgbImg, (hashSize, hashSize))
        rep = net.forward(rgbImg)
        return rep 

    rep = net.forward(alignedFace)
    return {
        "data": rep,
        "id": sum([2 ** i for (i, v) in enumerate(rep.flatten()) if v]),
        "indeces": -1
    }


if __name__=="__main__":
    image = cv2.imread("./images/queries/buddha.jpg")
    res = featurehash(image)
    # # print(res)

    image_2 = cv2.imread("./images/tendais/tendai.png")
    res_2 = featurehash(image_2)
    # print(res_2)

    print(feature_distance(res, res_2))

    hashmap = {
        res["id"] : {
            "data" : res["data"],
            "index": 1
        },
        res_2["id"] : {
            "data" : res["data"],
            "index": 2
        },
    }

    print(hashmap)

    # images = [image, image_2]
    my_dir = "./images/tendais"
    images = [cv2.imread(os.path.join(my_dir, i)) for i in os.listdir(my_dir)]
    
    cnn_face_detector = dlib.cnn_face_detection_model_v1("./models/dlib/mmod_human_face_detector.dat")

    exit()

    # win = dlib.image_window()
    hashes = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = dlib.load_rgb_image(f)
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        dets = cnn_face_detector(img, 1)
        '''
        This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
        These objects can be accessed by simply iterating over the mmod_rectangles object
        The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.
        
        It is also possible to pass a list of images to the detector.
            - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)

        In this case it will return a mmod_rectangless object.
        This object behaves just like a list of lists and can be iterated over.
        '''
        print("Number of faces detected: {}".format(len(dets)))
        for i, d in enumerate(dets):
            box = d.rect
            x = box.left()
            y = box.top()
            w = box.right() - x
            h = box.bottom() - y
            # get the detected face for hashing-hamming/openface embedding
            cropped = img[y:y+h, x:x+w]
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
            
            f_h = featurehash(cropped)
            
            if f_h is not None:
                hashes.append({
                    "image": cropped,
                    "hash" : f_h,
                })
        
    print(len(hashes))
    
    l = len(hashes)
    win_name = "images"
    cv2.namedWindow(win_name)
    for i in range(l):
        j = i + 1
        while j < l:
            i_1 = hashes[i]["image"]
            i_2 = hashes[j]["image"]
            b = (i_1.shape[0] + i_2.shape[0],
                    max(i_1.shape[1], i_2.shape[1]))

            bg = np.zeros(shape=(b[0] , b[1], 3), dtype=i_1.dtype)
            bg[:i_1.shape[0], :i_1.shape[1], :] = i_1
            bg[i_1.shape[0]:, :i_2.shape[1], :] = i_2

            dis = feature_distance(hashes[i]["hash"], hashes[j]["hash"])

            cv2.putText(bg, str(dis), (10, 10),cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
            cv2.imshow(win_name, bg)

            key = cv2.waitKey(5)
            if key == ord('q'):
                break

        # win.clear_overlay()
        # win.set_image(img)
        # win.add_overlay(rects)
        # dlib.hit_enter_to_continue()
    cv2.destroyWindow(win_name)