# -*- coding: utf-8 -*-
# 画像回転
# 画像読み込み -> 画像出力先指定 -> 回転画像生成・画像化
from cv2 import cv2
import numpy as np
import os
import shutil

mode = 'train'

def makeDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print('make directory ' + path)

""" 画像読み込み """
category_list = ['japanese_cedar', 'japanese_cypress', 'other_tree', 'not_tree']
for category in category_list:
    data_dir = './../../Data/trimed_img/randomsamplingdata_256to128/'+category+'/'
    #data_dir = './../forest_boundary/tree-species_discrimination/train_dataset/new_k-means/train/gamma-1000/train/' + category + '/'
    #data_dir = './../forest_boundary/tree-species_discrimination/train_dataset/original_in_oritani/in_oritani-k1000/train/'+ category +'/' # 入力画像フォルダのパス指定

    """ 画像出力先指定 """
    #output_dir = './../forest_boundary/tree-species_discrimination/train_dataset/new_k-means/train/gamma_rotate90-1000/' + category + '/'
    output_dir = './../../Data/trimed_img/rotate_randomsamplingdata_256to128/'+ category +'/' # 出力画像の保存先フォルダ指定
    shutil.copytree(data_dir, output_dir)
    
    image_num = 0

    image_dir = os.listdir(data_dir)
    image_num = len(image_dir)

    read_num = 0
    #read_max = 5 # read image max number 

    """ 回転画像生成・画像化 """
    angle_min = 90 
    angle = 90 # 角度指定
    scale = 1.0 
    angle_max = 360

    for image_n in image_dir:
        """
        if read_num >= read_max:
            break
        """
        image = cv2.imread(data_dir + image_n)
        #print('image : ', image)   
        for r in range(angle_min, angle_max, angle):
            size = tuple([image.shape[1], image.shape[0]])
            center = tuple([int(size[0]/2), int(size[1]/2)])
            rotation_matrix = cv2.getRotationMatrix2D(center, r, scale)
            rotation_image = cv2.warpAffine(image, rotation_matrix, size, flags = cv2.INTER_CUBIC)
            cv2.imwrite(output_dir + image_n.rstrip('.jpg') + '_rotate{rotate}.jpg'.format(rotate = r), rotation_image)
        read_num += 1
        
    print (category)
    print('rotate image : ', read_num)