import os
import sys
from cv2 import cv2
import numpy as np

import tensorflow as tf

import keras
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img

import low_value_image

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

save_dir = './../../Data/Report/'
model_path = './../../Data/reduce_othertree_Models/model_2250_0.92.h5'
original_dir = './../../../OrthoData/test/aerial/'
color_dict = [
                [0, 0, 255],
                [0, 128, 0],
                # [0, 255, 255],
                [128, 128, 128],
                [128, 128, 128],
                [0, 0, 0],
                [255, 255, 255]
                ]
white_rec = np.zeros((128, 128, 3), np.float32)
white_rec[:,:] = [255,255,255]
white_hist = cv2.calcHist([white_rec], [0], None, [256], [0, 256])
reject = 0.0

# load model
model = load_model(model_path)
size = 256
# resize_shape = (128, 128)

# create colormap
image_list = os.listdir(original_dir)
for image_n in image_list :
    #read images dir
    # imagename_without_ext = os.path.splitext(os.path.basename(image_n))[0]
    # low_value_image.makeDir(imagename_without_ext)
    image_path = original_dir + image_n
    src = img_to_array(load_img(image_path))
    src = src / 255.0
    if src is None:
        print(("Failed to load image file :"+image_path))
        continue
    height, width, channels = src.shape[:3]
    predicted_keeper = np.zeros((height, width, 3)) #4=[sugi, hinoki, other_tree, not_tree]
    num_inferences = np.zeros((height, width))
    colormap = np.zeros((height, width, channels))

    read_num = 1
    slide = 16
    for r_y in range(0, height, slide):
        X = []
        r = []
        ret_list = []
        if r_y + size >= height:
            break
        for r_x in range(0, width, slide):
            sys.stdout.write("\r load_n : %d" % read_num)
            sys.stdout.flush()
            if r_x + size >= width:
                break
            dst = src[r_y:r_y+size, r_x:r_x+size]   #image
            # resize_img = cv2.resize(dst, (resize_shape))

            # X.append(resize_img)   #image_list
            X.append(dst)

            r.append([r_y, r_x])    #coordinate_list

            # resize_img = resize_img * 255.0
            dst = dst * 255.0
            # img_hist = cv2.calcHist([resize_img], [0], None, [256], [0, 256])
            img_hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
            ret = cv2.compareHist(white_hist, img_hist, 0)
            ret_list.append(ret)    #background_list

            read_num += 1

        X = np.asarray(X)
        #save low value image
        predicted_value = model.predict(X, verbose=0)
        predicted_value = np.round(predicted_value, decimals=4)
        #low_value_image.trimmingImage(X, r, predicted_value, imagename_without_ext)
                
        for nth, r_n in enumerate(r):
            if ret_list[nth] < 1.0:
                # print(predicted_value)
                predicted_keeper[r_n[0]:r_n[0]+size, r_n[1]:r_n[1]+size] += predicted_value[nth]
                num_inferences[r_n[0]:r_n[0]+size, r_n[1]:r_n[1]+size] +=1
        # print(len(predicted_keeper))
        # print(len(predicted_keeper[0]))
        # print(len(predicted_keeper[0][0]))
        # print(predicted_keeper[0][0])

        # print(predicted_keeper.shape)
    # low_value_list = []
    for colormap_y in range(height):
        for colormap_x in range(width):
            sys.stdout.write("\r y : %d" % colormap_y)
            sys.stdout.flush()
            if np.max(predicted_keeper[colormap_y][colormap_x]) > 0:
                max_class = np.argmax(predicted_keeper[colormap_y][colormap_x])
                if np.max(predicted_keeper[colormap_y][colormap_x]) < low_value_image.low_value_line[max_class] * num_inferences[colormap_y][colormap_x]:
                    colormap[colormap_y][colormap_x] = low_value_image.boundary_color
                    low_value = low_value_image.createState(colormap_x, colormap_y, \
                        predicted_keeper[colormap_y][colormap_x], num_inferences[colormap_y][colormap_x])
                    # low_value_list.append(low_value)
                else:
                    colormap[colormap_y][colormap_x] = color_dict[np.argmax(predicted_keeper[colormap_y][colormap_x])]
            else:
                colormap[colormap_y][colormap_x] = color_dict[4]
    # np.array(low_value_list)
    # np.save('./../../Data/Report/lowvalue_'+imagename_without_ext, low_value_list)
    print('\nwrite colormap image -> png')
    image_name = image_n.replace('.jpg','')
    cv2.imwrite(save_dir + image_name + 'predicted_colormap'+'_0.05'+'.png', colormap)