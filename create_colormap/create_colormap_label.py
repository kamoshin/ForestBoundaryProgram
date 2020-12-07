import os
import sys
from cv2 import cv2
import numpy as np
import time
import pathlib

import tensorflow as tf

import keras
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

def makeDir(dir_path, mode='keep'):
    if mode == 'remove':
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    elif mode == 'keep':
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

save_dir = './../../../OrthoData/test/colormap/'
makeDir(save_dir)
aerial_dir = './../../../OrthoData/test/aerial/'
colormap_dir = './../../../OrthoData/test/colormap/'

slide = 16

makeDir(save_dir)

color_dict = [
                [0, 0, 255],
                [0, 128, 0],
                [0, 255, 255],
                [128, 128, 128],
                [0, 0, 0]
]

# create colormap
image_list = os.listdir(colormap_dir)
s_time = time.time()
for image_n in image_list:
    image_name = pathlib.Path(image_n)
    # img_s_time = time.time()
    #read images dir
    image_path = colormap_dir + image_n
    # src = img_to_array(load_img(image_path))
    # if src is None:
    #     print(("Failed to load image file :"+image_path))
    #     continue
    colormap = cv2.imread(image_path)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    height, width, channels = colormap.shape[:3]
    colormap_label = np.zeros((height, width, 4), dtype=int)

    for r_y in range(0, height, slide):
        if r_y + slide >= height:
            break
        for r_x in range(0, width, slide):
            sys.stdout.write("\r y : %d" % r_y)
            sys.stdout.flush()
            if r_x + slide >= width:
                break

            trimming_colormap = colormap[r_y:r_y+slide, r_x:r_x+slide]
            if np.sum(cv2.inRange(trimming_colormap, (255,0,0), (255,0,0)) == 255) == slide*slide:
                colormap_label[r_y:r_y+slide, r_x:r_x+slide] = np.array([1, 0, 0, 0])
            if np.sum(cv2.inRange(trimming_colormap, (0,128,0), (0,128,0)) == 255) == slide**2:
                colormap_label[r_y:r_y+slide, r_x:r_x+slide] = np.array([0, 1, 0, 0])
            if np.sum(cv2.inRange(trimming_colormap, (255,255,0), (255,255,0)) == 255) == slide**2:
                colormap_label[r_y:r_y+slide, r_x:r_x+slide] = np.array([0, 0, 1, 0])
            if np.sum(cv2.inRange(trimming_colormap, (128,128,128), (128,128,128)) == 255) == slide**2:
                colormap_label[r_y:r_y+slide, r_x:r_x+slide] = np.array([0, 0, 0, 1])

    np.save(save_dir+str(image_name.stem)+'_label', colormap_label)
    print(str(image_name.stem)+' colormap label saved')