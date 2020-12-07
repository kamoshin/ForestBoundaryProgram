import os
import sys
import cv2
import numpy as np
import time

import tensorflow as tf

import keras
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img

from keras.applications.densenet import preprocess_input

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

def makeDir(dir_path, mode='remove'):
    if mode == 'remove':
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    elif mode == 'keep':
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

save_dir = './../../Data/Report/DenseNet256/'
makeDir(save_dir)
model_path = './../../Data/Models/DenseNet256/model_best.h5'
original_dir = './../../../OrthoData/test/aerial/'

size = 256
reshape_size = 224
slide = 16

makeDir(save_dir)

color_dict = [
                [0, 0, 255],
                [0, 128, 0],
                [0, 255, 255],
                [128, 128, 128],
                [0, 0, 0]
                ]

# load model
model = load_model(model_path)

# create colormap
image_list = os.listdir(original_dir)
s_time = time.time()
for image_n in image_list :
    img_s_time = time.time()
    #read images dir
    image_path = original_dir + image_n
    src = img_to_array(load_img(image_path))
    white_in_image = cv2.inRange(src, (255,255,255), (255,255,255))
    src = preprocess_input(src)
    
    if src is None:
        print(("Failed to load image file :"+image_path))
        continue
    print(src.shape)
    height, width, channels = src.shape[:3]
    predicted_keeper = np.zeros((height, width, 4))
    colormap = np.zeros((height, width, channels))

    for r_y in range(0, height, slide):
        X = []
        r = []
        if r_y + size >= height:
            break
        for r_x in range(0, width, slide):
            sys.stdout.write("\r y : %d" % r_y)
            sys.stdout.flush()
            if r_x + size >= width:
                break

            dst = src[r_y:r_y+size, r_x:r_x+size]
            dst = cv2.resize(dst, (reshape_size, reshape_size))
            #dst = preprocess_input(dst)

            if np.sum(white_in_image[r_y:r_y+size, r_x:r_x+size] == 255) == 0:
                X.append(dst)
                r.append([r_y, r_x])

        if len(X) >= 1:
            X = np.asarray(X)
            predicted_value = model.predict(X, verbose=0)
            for nth, r_n in enumerate(r):
                predicted_keeper[r_n[0]:r_n[0]+size, r_n[1]:r_n[1]+size] += predicted_value[nth]

    bg_map = cv2.inRange(predicted_keeper, (0,0,0,0), (0,0,0,0))

    for n in range(len(color_dict) - 1):
        colormap[np.where(np.argmax(predicted_keeper, axis=2) == n)] = color_dict[n]
    colormap[np.where(bg_map == 255)] = color_dict[len(color_dict)-1]
    print('\nwrite colormap image -> png')
    image_name = image_n.replace('.jpg','')
    cv2.imwrite(save_dir + image_name + 'predicted_colormap.png', colormap)

    print("time : {0:.1f}[sec]".format(time.time()-img_s_time))

print("time : {0:.1f}[sec]".format(time.time()-s_time))