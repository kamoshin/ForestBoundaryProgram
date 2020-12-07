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

save_dir = './../../Data/Report/train/'
makeDir(save_dir)
model_path = './../../Data/Models/few_data32/model_300_0.61.h5'
original_dir = './../../../OrthoData/train/aerial/'
colormap_dir = './../../../OrthoData/train/colormap/'

size = 32
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
for image_n in image_list:
    image_name = pathlib.Path(image_n)
    img_s_time = time.time()
    #read images dir
    image_path = original_dir + image_n
    src = img_to_array(load_img(image_path))
    src = src / 255.0
    if src is None:
        print(("Failed to load image file :"+image_path))
        continue
    print(src.shape)
    height, width, channels = src.shape[:3]
    predicted_keeper = np.zeros((height, width, 4))
    predicted_num = np.zeros((height, width, 4))   #推論回数
    # colormap = np.zeros((height, width, channels))
    colormap = cv2.imread(colormap_dir+"colormap/"+image_n+"_colormap.png")
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    colormap_label = np.zeros((height, width, 4))

    for r_y in range(0, height, slide):
        X = []
        r = []
        if r_y + size >= height:
            break
        for r_x in range(0, width, slide):
            sys.stdout.write("\r y : %d" % r_y)
            sys.stdout.flush()
            test_list_label = 0
            if r_x + size >= width:
                break

            dst = src[r_y:r_y+size, r_x:r_x+size]

            white_in_image = cv2.inRange(dst, (1,1,1), (1,1,1)) #画像のwhite部分(背景)が含まれていてもデータとして取られてしまうのを防ぐための変数

            if np.sum(white_in_image == 255) == 0:
                X.append(dst)
                # X.append(dst)
                r.append([r_y, r_x])

        if len(X) >= 1:
            X = np.asarray(X)
            predicted_value = model.predict(X, verbose=0)
            for nth, r_n in enumerate(r):
                color_count = []
                predicted_keeper[r_n[0]:r_n[0]+size, r_n[1]:r_n[1]+size] += predicted_value[nth]
                predicted_num[r_n[0]:r_n[0]+size, r_n[1]:r_n[1]+size] +=1
                trimming_colormap = colormap[r_n[0]:r_n[0]+slide, r_n[1]:r_n[1]+slide]
                color_count.append(np.sum(cv2.inRange(trimming_colormap, (255,0,0), (255,0,0)) == 255))
                color_count.append(np.sum(cv2.inRange(trimming_colormap, (0,128,0), (0,128,0)) == 255))
                color_count.append(np.sum(cv2.inRange(trimming_colormap, (255,255,0), (255,255,0)) == 255))
                color_count.append(np.sum(cv2.inRange(trimming_colormap, (128,128,128), (128,128,128)) == 255))
                colormap_label[r_n[0]:r_n[0]+size, r_n[1]:r_n[1]+size] = np.argmax(color_count)
                


                
            
    print(predicted_keeper[0][0])
    print(predicted_num[0][0])
    predicted_keeper_average = np.divide(predicted_keeper, predicted_num, \
        out=np.zeros_like(predicted_keeper), where=predicted_num!=0)
    print(predicted_keeper_average[0][0])
    print(str(image_name.stem)+' predicted keeper average saved')
    np.save(save_dir+str(image_name.stem)+'_'+str(size)+'_predicted_keeper', predicted_keeper_average)