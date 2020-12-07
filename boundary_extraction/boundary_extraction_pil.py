import os
import sys
from PIL import Image
import numpy as np
import csv
from sklearn.metrics import classification_report
import pandas as pd

from cv2 import cv2

import keras
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image

from keras.applications.densenet import preprocess_input

# img_dir = './../../Data/Report/DenseNet/32+128/32+128/'
# img_path = img_dir+'yuusenji_Dense32+128_pred_colormap.png'
# img_dir = './../../Data/Report/DenseNet/32+128/32+128/'
# img_path = img_dir+'ushitsu_Dense32+128_pred_colormap.png'
img_dir = './../../../OrthoData/test/colormap/'
img_path = img_dir+'yuusenji_colormap.png'

# colormap = cv2.imread(img_path)
# colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
colormap = Image.open(img_path)
colormap = colormap.convert('RGB')

width, height = colormap.size[:2]
boundary_pointer = np.zeros((height, width))
boundary_map = np.zeros((height, width, 3))

check_size = 32
slide = 16

for y in range(0, height, slide):
    if y+check_size > height:
        break
    sys.stdout.write('\r y_1:%d' % y)
    sys.stdout.flush()
    for x in range(0, width, slide):
        if x+check_size > width:
            break
        first_r, first_g, first_b = colormap.getpixel((x, y))
        color_count = [0,0,0,0]
        for r_y in range(y, y+check_size):
            for r_x in range(x, x+check_size):
                r,g,b = colormap.getpixel((r_x,r_y))
                if r==255 and g==0 and b==0:
                    color_count[0] = 1
                elif r==0 and g==128 and b==0:
                    color_count[1] = 1
                elif r==255 and g==255 and b==0:
                    color_count[2] = 1
                elif r==128 and g==128 and b==128:
                    color_count[3] = 1
        color_count = [color for color in color_count if color == 1]
        if len(color_count) < 2:
            boundary_map[y:y+check_size, x:x+check_size] = np.array([first_r, first_g, first_b])
        elif len(color_count) >=2:
            boundary_pointer[y:y+check_size, x:x+check_size] = 1

boundary_map[np.where(boundary_pointer == 1)] = np.array([160, 32, 255])

boundary_map = np.array(boundary_map, dtype = 'float32')
boundary_map = cv2.cvtColor(boundary_map, cv2.COLOR_BGR2RGB)
cv2.imwrite(img_dir+'yuusenji_boundarymap.png', boundary_map)
