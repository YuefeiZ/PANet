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
import json
import yaml
from PIL import Image

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
    print("cuiwei***********************")

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
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 256)  # anchor side in pixels

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

class ChromsomeDataset(utils.Dataset):
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def get_obj(self, path):
        s = path.split('\\')
        path1 = 'H:\\Data\\chromosome\\img_json\\'
        s = s[4]
        img_num = s.split('.')
        img_num = img_num[0]
        imgfile = path1 + img_num + '_json\\' + 'label.png'
        img = cv2.imread(imgfile)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
        image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)
        return n

    def get_obj_num(self, path):
        s = path.split('\\')
        path1 = 'H:\\Data\\chromosome\\json\\'
        s = s[4]
        img_num = s.split('.')
        img_num = img_num[0]
        annotations = json.load(open(path1 + img_num + '.json'))

        annotations = list(annotations.values())  # don't need the dict keys
        # for a in annotations:

        n = len(annotations[2])
        return n


    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    def draw_mask(self, mask, path):
        s = path.split('\\')
        path1 = 'H:\\Data\\chromosome\\json\\'
        s = s[4]
        img_num = s.split('.')
        img_num = img_num[0]
        with open(path1 + img_num + '.json', "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
            a = data['shapes']

            for i in range(len(a)):
                points_x = []
                points_y = []
                for x, y in a[i]['points']:
                    points_x.append(x)
                    points_y.append(y)
                    # print(x)
                    # print(y)
                # print(points_x)
                # print(points_y)
                #print(b['points'])
                X = np.array(points_x)
                Y = np.array(points_y)
                rr, cc = skimage.draw.polygon(Y, X)
                mask[rr, cc, i] = 1
        return mask


    def load_shapes(self, count, img_floder,mask_floder,imglist,dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "impurity")
        #random.shuffle(imglist)
        for i in range(count):
            filestr = imglist[i].split(".png")[0]
            mask_path = mask_floder + "\\" + filestr + ".png"
            yaml_path = dataset_root_path +'img_json\\' + filestr + "_json\\info.yaml"
            image = Image.open(dataset_root_path +'img_json\\'+ filestr + "_json\\img.png")
            w, h = image.size
            self.add_image("shapes", image_id=i, path=img_floder + "\\" + imglist[i],
                           width=w, height=h, mask_path=mask_path, yaml_path=yaml_path)


    def load_mask(self, image_id):
          """Generate instance masks for shapes of the given image ID.
          """
          global iter_num
          #print("self.image_info", self.image_info)
          info = self.image_info[image_id]
          count = self.get_obj_num(info['path']) # number of object
          #print(info)
          #img = Image.open(info['mask_path'])
         # global count_num
         # count_num = count_num + 1
         # print(count_num)
          mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
          mask = self.draw_mask(mask, info['path'])
          #for i in range(count):
          #    mask1 = mask[:,:,i]
          #    cv2.imshow("s", mask1)
          #    cv2.waitKey()

          labels=[]
          labels=self.from_yaml_get_class(image_id)
          labels_form=[]
          for i in range(count):
                labels_form.append("impurity")

          class_ids = np.array([self.class_names.index(s) for s in labels_form])
          #print(class_ids)
          return mask.astype(np.bool), class_ids.astype(np.int32)

    def get_ax(rows=1, cols=1, size=8):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.

        Change the default size attribute to control the size
        of rendered images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax

path ='H:\\Data\\chromosome\\test\\'
name = 'testimage.txt'
#with open(path + name) as f:
#    for line in f:
#    #for i in range(len(imglist)):
#        l = line.split('\n')
#        imagpath = path + l[0] + '.png'
#        #imagpath = path1 + imglist[i]
#        image = skimage.io.imread(imagpath)
#
#        # Run detection
#        results = model.detect([image], verbose=1)
#
#        # Visualize results
#        r = results[0]
#        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                        class_names, r['scores'])
#        #visualize.display_instances1(image, r['rois'], r['masks'], r['class_ids'],
#         #                            class_names, r['scores'], l[0])
dataset_root_path="H:\\Data\\chromosome_refine\\"
img_floder = dataset_root_path + "test"
mask_floder = dataset_root_path + "mask"
#yaml_floder = dataset_root_path
imglist = os.listdir(img_floder)
count = len(imglist)

count_num = 0
dataset_val = ChromsomeDataset()
dataset_val.load_shapes(count, img_floder, mask_floder, imglist,dataset_root_path)
dataset_val.prepare()

image_ids = np.random.choice(dataset_val.image_ids, 100)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))