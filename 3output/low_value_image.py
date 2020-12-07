import os
import shutil
from cv2 import cv2
import numpy as np

low_value_dir = './../../Data/Low_value_image/'
low_value_line = (0.5, 0.5, 0.5, 0.5)
boundary_color = (255, 0, 0)

def makeDir(image_name):
    if os.path.exists(low_value_dir+image_name+'/'):
        shutil.rmtree(low_value_dir+image_name+'/')
    os.mkdir(low_value_dir+image_name+'/')
    print('make directory ' + image_name)

def trimmingImage(image_list, coordinate_list, value_list, image_name):
    for (image, coordinate, value) in zip(image_list, coordinate_list, value_list):
            if np.max(value) < low_value_line:
                image = image * 255.0
                cv2.imwrite(low_value_dir+str(image_name)+'/'+str(coordinate[0])+"_"+str(coordinate[1])+".png", image)

def createState(x, y, sum_value, num):
    point = [x, y]
    low_value_state = np.array([point, sum_value, num])
    return low_value_state