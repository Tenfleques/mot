#!/bin/python

from hashing import convert_hash, hammingDistance, dhash
from imutils import paths
import argparse
import pickle
import vptree
import cv2
import os 
import timeit
import time 

# USAGE
# python index_images.py --images ../images/101_ObjectCategories --tree ../data/vptree.pickle --hashes ../data/hashes.pickle

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, type=str,
    help="path to input directory of images")
ap.add_argument("-t", "--tree", required=True, type=str,
    help="path to output VP-Tree")
ap.add_argument("-a", "--hashes", required=True, type=str,
    help="path to output hashes dictionary")
args = vars(ap.parse_args())


# grab the paths to the input images and initialize the dictionary
# of hashes
def createVPTree(images, hashfile, treefile):
    imagePaths = list(paths.list_images(images))
    hashes = {}
    for (i, imagePath) in enumerate(imagePaths):

        # print("[INFO] processing image {}/{}".format(i + 1,
            # len(imagePaths)))
        image = cv2.imread(imagePath)

        # compute the hash for the image and convert it
        h = dhash(image)
        h = convert_hash(h)

        # update the hashes dictionary
        l = hashes.get(h, [])
        l.append(imagePath)
        hashes[h] = l

    # build the VP-Tree
    print("[INFO] building VP-Tree...")
    points = list(hashes.keys())
    tree = vptree.VPTree(points, hammingDistance)

    # serialize the VP-Tree to disk
    print("[INFO] serializing VP-Tree...")
    f = open(treefile, "wb")
    f.write(pickle.dumps(tree))
    f.close()

    # serialize the hashes to dictionary
    print("[INFO] serializing hashes...")
    f = open(hashfile, "wb")
    f.write(pickle.dumps(hashes))
    f.close()

def main():
    createVPTree(args["images"], args["hashes"], args["tree"])
    return 0

start = time.time()
# timeit.timeit('main()', 10000)
main()
end = time.time()

print("time of execution: {}".format(end - start))