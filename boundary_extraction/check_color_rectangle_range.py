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
img_dir = './../../Data/Report/DenseNet/32+128/32+128/'
img_path = img_dir+'yuusenji_Dense32+128_pred_colormap.png'

colormap = cv2.imread(img_path)
colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
# img1 = Image.open(img_path)
# img1 = img1.convert('RGB')

height, width, channels = colormap.shape[:3]
# boundary_pointer = np.zeros((height, width))
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
        pixel_color = colormap[y][x]
        colormap_rectangle = colormap[y:y+check_size, x:x+check_size]
        color_count = [0,0,0,0]
        for r_y in range(y, y+check_size):
            for r_x in range(x, x+check_size):
                if np.all(colormap[r_y][r_x] == [255, 0, 0]):
                    color_count[0] = 1
                elif np.all(colormap[r_y][r_x] == [0, 128, 0]):
                    color_count[1] = 1
                elif np.all(colormap[r_y][r_x] == [255, 255, 0]):
                    color_count[2] = 1
                elif np.all(colormap[r_y][r_x] == [128, 128, 128]):
                    color_count[3] = 1
        color_count = [color for color in color_count if color == 1]
        if len(color_count) < 2:
            boundary_map[y:y+check_size, x:x+check_size] = pixel_color
        elif len(color_count) >=2:
            boundary_map[y:y+check_size, x:x+check_size] = np.array([160, 32, 255])

# for y in range(16, height, slide):
#     sys.stdout.write('\r y_2:%d' % y)
#     sys.stdout.flush()
#     for x in range(16, width, slide):
#         color_count = 0
#         pixel_color = colormap[y][x]
#         colormap_rectangle = colormap[y:y+check_size, x:x+check_size]
#         for r_y in range(check_size):
#             for r_x in range(check_size):
#                 if np.all(colormap[r_y][r_x] == [255, 0, 0]):
#                     color_count += 1
#                 elif np.all(colormap[r_y][r_x] == [0, 128, 0]):
#                     color_count += 1
#                 elif np.all(colormap[r_y][r_x] == [255, 255, 0]):
#                     color_count += 1
#                 elif np.all(colormap[r_y][r_x] == [128, 128, 128]):
#                     color_count += 1
#         if color_count >= 2:
#             boundary_pointer[y:y+check_size, x:x+check_size] = 1
#         else:
#             boundary_map[y:y+check_size, x:x+check_size] = pixel_color

# for y in range(height):
#     for x in range(width):
#         if boundary_pointer[y][x] == 1:
#             boundary_map[y][x] = np.array([160, 32, 255])

boundary_map = np.array(boundary_map, dtype = 'float32')
boundary_map = cv2.cvtColor(boundary_map, cv2.COLOR_BGR2RGB)
cv2.imwrite(img_dir+'yuusenji32+128_boundarymap.png', boundary_map)
