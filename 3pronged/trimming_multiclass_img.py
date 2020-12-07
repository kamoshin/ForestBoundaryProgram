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

def makeDir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

# Target RGB color of map_image    
def selectColor(target_name):
    select_color = (0,0,0)    
    if target_name == "japanese_cedar": # スギ
        select_color = (255,0,0)
    elif target_name == "japanese_cypress": # ヒノキ
        select_color = (0,128,0)
    elif target_name == "other_tree": # 他樹種       
        select_color = (255,255,0)
    elif target_name == "not_tree":    
        select_color = (128,128,128) # 樹種以外
    else :
        print("error select color")
    
    return select_color

#image_size = (7248,4849)
# sampling_image_size = (128,128)
sampling_image_size = (256, 256)

orig_dir = './../../../OrthoData/test/'

save_dir = "./../../Data/trimed_img/test_multiclass_image/"
#save_dir = "./../../Dataset/trimed_test_dataset/"
#save_dir = "./../../Dataset/trimed_ushitsu_dataset/"
#save_dir = "./../../Dataset/trimed_yuusenji_dataset/"
makeDir(save_dir)

def main():
    photo_name_list = ['ushitsu', 'yuusenji']
    slide = 64

    for photo_name in photo_name_list:
        # read image
        map_image = cv2.imread(orig_dir+"colormap/"+photo_name+"_colormap.png")
        aerial_image = cv2.imread(orig_dir+"aerial/"+photo_name+".jpg")
        map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
        aerial_image = cv2.cvtColor(aerial_image, cv2.COLOR_BGR2RGB)
        makeDir(save_dir)             
        height, width, channels = aerial_image.shape[:3]

        save_image_num = 0
        w = sampling_image_size[0]
        h = sampling_image_size[1]
        # for loop, image size ( refer : https://techacademy.jp/magazine/15828 )
        for x in range(0,width,slide):
            for y in range(0,height,slide):
                # trimming ( refer : https://algorithm.joho.info/programming/python/opencv-roi-py/ )
                trimming_map_image = map_image[y:y+h, x:x+w]
                
                class_count = 0
                red_in_image = cv2.inRange(trimming_map_image, (255,0,0), (255,0,0))
                green_in_image = cv2.inRange(trimming_map_image, (0,128,0), (0,128,0))
                yellow_in_image = cv2.inRange(trimming_map_image, (255,255,0), (255,255,0))
                gray_in_image = cv2.inRange(trimming_map_image, (128,128,128), (128,128,128))

                red_rate = round(np.sum(red_in_image == 255)//(w*h), 1)
                green_rate = round(np.sum(red_in_image == 255)//(w*h), 1)
                yellow_rate = round(np.sum(red_in_image == 255)//(w*h), 1)
                gray_rate = round(np.sum(red_in_image == 255)//(w*h), 1)

                if np.sum(red_in_image == 255) >= w*h//4:
                    class_count += 1
                if np.sum(green_in_image == 255) >= w*h//4:
                    class_count += 1
                if np.sum(yellow_in_image == 255) >= w*h//4:
                    class_count += 1
                if np.sum(gray_in_image == 255) >= w*h//4:
                    class_count += 1
                
                black_in_image = cv2.inRange(trimming_map_image, (0,0,0), (0,0,0)) #画像の黒い部分(背景)が含まれていてもデータとして取られてしまうのを防ぐための変数
                      
                if np.sum(black_in_image == 255) == 0 and class_count >= 2:
                    # trimming
                    trimming_color_image = aerial_image[y:y+h, x:x+w]
                    
                    # save color image
                    point = "_x"+str(x)+"y"+str(y)
                    save_name = save_dir+photo_name+"/"+photo_name+"_"+str(save_image_num)+point+\
                                str(red_rate)+"_"+str(green_rate)+"_"+str(yellow_rate)+"_"+str(gray_rate)+".jpg"
                    save_image = cv2.cvtColor(trimming_color_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_name,save_image)
                    save_image_num += 1
                    
                    
        print("{} image num :{}".format(photo_name, save_image_num))

if __name__ == '__main__':
    main()