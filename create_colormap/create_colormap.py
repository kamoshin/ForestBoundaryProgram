import os
import sys
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img


# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True
# keras.backend.set_session(tf.Session(config=config))

save_dir = './../../Data/Report/'
model_path = './../../Data/model/model_94_0.60.h5'
original_dir = './../../../Original_Dataset/test/aerial/'
color_dict = {
                "red": [0, 0, 255],
                "green": [0, 128, 0],
                "yellow": [0, 255, 255],
                "gray": [128, 128, 128],
                "black": [0, 0, 0],
                "white": [255, 255, 255]
            }
white_rec = np.zeros((128, 128, 3), np.float32)     #縦128、横128、色3の配列を用意
white_rec[:,:] = [255,255,255]      #全ての要素を白に
white_hist = cv2.calcHist([white_rec], [0], None, [256], [0, 256])
reject = 0.0

# load model
model = load_model(model_path)
image_shape = (128, 128, 3)

# create colormap
image_list = os.listdir(original_dir)
for image_n in image_list:
    #read images dir
    image_path = original_dir + image_n
    src = img_to_array(load_img(image_path))
    src = src / 255.0
    if src is None:
        print(("Failed to load image file :"+image_path))
        continue
    height, width, channels = src.shape[:3]
    size = 128
    colormap = np.zeros((height, width, channels), np.uint8)

    read_num = 1
    slide = 16
    for r_y in range(0, height, slide):
        if r_y + size >= height:
            break
        X = []
        r = []
        ret_list = []
        for r_x in range(0, width, slide):
            sys.stdout.write("\r load_n : %d" % read_num)
            sys.stdout.flush()
            test_list_label = 0
            if r_x + size >= width:
                break
            dst = src[r_y:r_y+size, r_x:r_x+size]

            X.append(dst)

            r.append([r_y, r_x])

            dst = dst * 255.0
            img_hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
            ret = cv2.compareHist(white_hist, img_hist, 0)
            ret_list.append(ret)

            read_num += 1

        X = np.asarray(X)
        predicted_value = model.predict(X, batch_size=1024, verbose=0)
        for nth, r_n in enumerate(r):
            if ret_list[nth] < 1.0:
                if np.max(predicted_value[nth]) <= reject:
                    colormap[r_n[0]:r_n[0]+size, r_n[1]:r_n[1]+size] = color_dict["white"]
                else:
                    colormap[r_n[0]:r_n[0]+size, r_n[1]:r_n[1]+size] = color_dict[np.argmax(predicted_value[nth])]
            else:
                colormap[r_n[0]:r_n[0]+size, r_n[1]:r_n[1]+size] = color_dict["black"]

    print('\nwrite colormap image -> png')
    image_name = image_n.replace('.jpg','')
    cv2.imwrite(save_dir + image_name + 'predicted_colormap.png', colormap)