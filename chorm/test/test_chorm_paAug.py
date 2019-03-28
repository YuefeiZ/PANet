# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
import re
from mrcnn.config import Config
from datetime import datetime

from quiver_engine import server
# Root directory of the project
ROOT_DIR ='I:\\pythoncode\\Mask_RCNN-master'
sys.path.append(ROOT_DIR)  # To find local version of the library
# Import Mask RCNN
  # To find local version of the library
from mrcnn import utils
import mrcnn.panetmodelAug as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.coco import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs\\chorm_paaug20181220T1148\\mask_rcnn_chorm_paaug_0015.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("yuefei***********************")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 2  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16,32 , 64 , 128 , 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TTRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 540

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20

#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'impurity']
# Load a random image from the images folder



path ='H:\\Data\\chromosome\\test\\'
name = 'testimage.txt'
with open(path + name) as f:
    for line in f:
    #for i in range(len(imglist)):
        l = line.split('\n')
        imagpath = path + l[0] + '.png'
        #imagpath = path1 + imglist[i]
        image = skimage.io.imread(imagpath)

        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                        class_names, r['scores'])
        #visualize.display_instances1(image, r['rois'], r['masks'], r['class_ids'],
        #                            class_names, r['scores'], l[0])

