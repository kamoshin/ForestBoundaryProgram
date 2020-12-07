import numpy as np
import os
from cv2 import cv2

slide = 16
size1 = 256
size2 = 32
img_list = ['bamba', 'bodai', 'kurikara', 'matii', 'nashitani', 'oritani',\
            'rokuro', 'takemata', 'togi', 'urii', 'yoshikuraB']
dir = './../../Data/Report/train/'
orig_dir = './../../../OrthoData/train/'
path_list = os.listdir(dir)
for photo_name in img_list:
    pred_value = np.load(dir+photo_name)
    height, width, channels = pred_value.shape[:3]
    pred_value_list =[]
    label_list = []
    map_image = cv2.imread(orig_dir+"colormap/"+photo_name+"_colormap.png")
    map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
    for y in range(0, height, slide):
        for x in range(0, width, slide):
            trimming_map_image = map_image[y:y+slide, x:x+slide]
            black_in_image = cv2.inRange(trimming_map_image, (0,0,0), (0,0,0))

            #add pred_value

    print(pred_value[8000][8000])
    print(pred_value[7999][8000])
    exit()

