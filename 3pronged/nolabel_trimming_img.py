# -*- coding: utf-8 -*-
#津崎さんの同名のプログラムを、黒い部分(背景)をデータとして取らないように少し変更。以降の trimming_image_X はこれを基本としている(林悠月)
"""
Created on Fri Nov 16 21:25:18 2018

@author: kawasaki
"""
import os
from cv2 import cv2
import numpy as np

import matplotlib.pyplot as plt
#---Matlab function---
def showImagePLT(im):
    plt.imshow(im)
    plt.show()

def makeDir(dir_path, mode='keep'):
    if mode == 'remove':
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        print('make directory '+dir_path)
    elif mode == 'keep':
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print('make directory '+dir_path)

sampling_image_size = (256, 256)
slide = 64

orig_dir = './../../../../OrthoData/test/'

save_dir = './../../../Data/trimed_img/nolabel_test/'
makeDir(save_dir)

def main():
    photo_name_list = ['ushitsu', 'yuusenji']

    for photo_name in photo_name_list:
        # read image
        aerial_image = cv2.imread(orig_dir+"aerial/"+photo_name+".jpg")
        aerial_image = cv2.cvtColor(aerial_image, cv2.COLOR_BGR2RGB)
        makeDir(save_dir+photo_name)
                
        # show image    
        #showImagePLT(map_image)
        #showImagePLT(aerial_image)               
        
        height, width, channels = aerial_image.shape[:3]
        
        save_image_num = 0
        w = sampling_image_size[0]
        h = sampling_image_size[1]
        # for loop, image size ( refer : https://techacademy.jp/magazine/15828 )
        for x in range(0,width,slide):
            for y in range(0,height,slide):
                # trimming ( refer : https://algorithm.joho.info/programming/python/opencv-roi-py/ )
                trimming_aerial_image = aerial_image[y:y+h, x:x+w]                    
                # save color image
                point = "_x"+str(x)+"y"+str(y)
                save_name = save_dir+photo_name+"/"+photo_name+"_"+str(save_image_num)+point+".jpg"
                save_image = cv2.cvtColor(trimming_aerial_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_name,save_image)
                save_image_num += 1
                    
        print("{} image num :{}".format(photo_name, save_image_num))

if __name__ == '__main__':
    main()