import numpy as np
import os
from cv2 import cv2

image_list = ['yuusenji']
report_dir = './../../Data/Report/DenseNet256/'

size = 256

color_dict = [
                [0, 0, 255],
                [0, 128, 0],
                [0, 255, 255],
                [128, 128, 128],
                [0, 0, 0]
                ]

for image_n in image_list:
    # pred_keeper32 = np.load('./../../Data/Report/few_data32/'+image_n+'_32_predicted_keeper.npy')
    pred_keeper_num256 = np.load('./../../Data/Report/few_data/'+image_n+'_256_predicted_num.npy')
    pred_keeper256 = np.load(report_dir+image_n+'_256_predicted_keeper.npy')
    print(pred_keeper256[3200][3200])
    print(pred_keeper_num256[3200][3200])
    pred_keeper_average = pred_keeper256 / pred_keeper_num256
    print(pred_keeper_average[3200][3200])
    np.save(report_dir+str(image_n)+'_'+str(size)+'_predicted_keeper_ave', pred_keeper_average)

#     print("time : {0:.1f}[sec]".format(time.time()-img_s_time))

# print("time : {0:.1f}[sec]".format(time.time()-s_time))
