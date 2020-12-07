import os
import numpy as np
import sys

import keras
from keras.models import load_model
from cv2 import cv2

np.set_printoptions(linewidth=100000)

l_size = 256
m_size = 128
s_size = 64
ss_size = 32
slide = 16

#read images dir
# image_list = ['bamba', 'kurikara', 'matii', 'takemata', 'togi', 'urii']
image_list = ['ushitsu', 'yuusenji']
pred_value_dir = './../../../../Data/pred_value/DenseNet'
# pred_value_alex_dir = './../../../../Data/pred_value/AlexNet'
save_dir = './../../../../Data/Report/DenseNet/'+str(ss_size)+'+'+str(s_size)+'+'+str(m_size)+'+'+str(l_size)+'/NN/'
model_path = './../../../../Data/Models/integ_pred_value/model_1500_0.81.h5'

model = load_model(model_path)

# print(model.predict(np.array([[0.8, 0.2, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0, 0.4, 0.3, 0.2, 0.1]])))
# exit()

color_dict = [
                [0, 0, 255],
                [0, 128, 0],
                [0, 255, 255],
                [128, 128, 128],
                [0, 0, 0]
                ]

for image_n in image_list:
    print(image_n)
    read_num = 0
    print('lerge_size loading...')
    l_pred = np.load(pred_value_dir+str(l_size)+'/'+str(image_n)+'_'+str(l_size)+'_predicted_keeper.npy')
    print('middle_size loading...')
    m_pred = np.load(pred_value_dir+str(m_size)+'/'+str(image_n)+'_'+str(m_size)+'_predicted_keeper.npy')
    print('small_size loading...')
    s_pred = np.load(pred_value_dir+str(s_size)+'/'+str(image_n)+'_'+str(s_size)+'_predicted_keeper.npy')
    print('most_small_size loading...')
    ss_pred = np.load(pred_value_dir+str(ss_size)+'/'+str(image_n)+'_'+str(ss_size)+'_predicted_keeper.npy')
    height, width, channels = l_pred.shape[:3]
    colormap = np.zeros((height, width, 3)) 
    predicted_keeper = np.zeros((height, width, 4))

    for r_y in range(0, height, slide):
        if r_y + slide >= height:
            break
        for r_x in range(0, width, slide):
            X = []
            sys.stdout.write("\r y : %d" % r_y)
            sys.stdout.flush()
            if r_x + slide >= width:
                break
            pred_value_set = []
            pred_value_set.extend(l_pred[r_y][r_x])
            pred_value_set.extend(m_pred[r_y][r_x])
            pred_value_set.extend(s_pred[r_y][r_x])
            pred_value_set.extend(ss_pred[r_y][r_x])
            if not any(np.isnan(pred_value_set)):
                X.append(pred_value_set)
                predicted_pred_value = model.predict(np.array(X))
                colormap[r_y:r_y+slide, r_x:r_x+slide] = color_dict[np.argmax(predicted_pred_value)]
            else:
                colormap[r_y][r_x] = color_dict[4]
    print('\nwrite colormap image -> png')
    image_name = image_n.replace('.jpg','')
    cv2.imwrite(save_dir + image_name + 'pred_colormap.png', colormap)

print('end')
