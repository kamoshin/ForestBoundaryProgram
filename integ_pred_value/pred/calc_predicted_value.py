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

from keras.applications.densenet import preprocess_input

Image.MAX_IMAGE_PIXELS = 1000000000

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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

# size = 256
size = 32
slide = 16

save_dir = './../../../Data/pred_value/DenseNet'+str(size)+'/'
# save_dir = './../../Data/Report/few_data/'
makeDir(save_dir)
model_path = './../../../Data/Models/DenseNet'+str(size)+'/model_best.h5'
# model_path = './../../../Data/Models/few_data/model_1480_0.84.h5'
original_dir = './../../../../OrthoData/train/aerial/'

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
# image_list = ['bamba.jpg', 'kurikara.jpg', 'matii.jpg', 'takemata.jpg', 'togi.jpg', 'urii.jpg']
s_time = time.time()
for image_n in image_list:
    image_name = pathlib.Path(image_n)
    img_s_time = time.time()
    #read images dir
    image_path = original_dir + image_n
    src = img_to_array(load_img(image_path))
    white_in_image = cv2.inRange(src, (255,255,255), (255,255,255))
    src = preprocess_input(src)
    # src = src / 255.0
    if src is None:
        print(("Failed to load image file :"+image_path))
        continue
    height, width, channels = src.shape[:3]
    predicted_keeper = np.zeros((height, width, 4))
    predicted_num = np.zeros((height, width, 1))   #推論回数

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
            # dst = cv2.resize(dst, (reshape_size, reshape_size))
            
            if np.sum(white_in_image[r_y:r_y+size, r_x:r_x+size] == 255) == 0:
                X.append(dst)
                r.append([r_y, r_x])

        if len(X) >= 1:
            X = np.asarray(X)
            predicted_value = model.predict(X, verbose=0)
            for nth, r_n in enumerate(r):
                predicted_keeper[r_n[0]:r_n[0]+size, r_n[1]:r_n[1]+size] += predicted_value[nth]
                predicted_num[r_n[0]:r_n[0]+size, r_n[1]:r_n[1]+size] +=1

    predicted_keeper_average = predicted_keeper / predicted_num
    print(str(image_name.stem)+' predicted keeper average saved')
    np.save(save_dir+str(image_name.stem)+'_'+str(size)+'_predicted_keeper', predicted_keeper_average)
    cv2.imwrite(save_dir+'predicted_colormap_ave.png', predicted_keeper_average)
