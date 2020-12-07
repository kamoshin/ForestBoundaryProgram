import os
from cv2 import cv2
import shutil
from glob import glob
import numpy as np

# def makeDir(dir_path):
#     if not os.path.isdir(dir_path):
#         os.mkdir(dir_path)

resize_img = (128,128)

# data_dir = "./../../Dataset/train/"
# data_dir = "./../../Data/trimed_img/train/"
# save_dir = "./../../Data/trimed_img/train_256to128/"
data_dir = "./../../Data/trimed_img/randomsamplingdata_256/"
save_dir = "./../../Data/trimed_img/randomsamplingdata_256to128/"

if os.path.exists(save_dir):
    #print('!!!save_dir name is duplicated!!!')
    #exit()
    shutil.rmtree(save_dir)
# os.mkdir(save_dir)
os.makedirs(save_dir, exist_ok=True)
# makeDir(save_dir)

image_name_list = ['japanese_cypress', 'japanese_cedar', 'other_tree', 'not_tree']

for image_name in image_name_list:

    image_path_list = glob(data_dir + image_name +'/*.jpg')
    # print(image_path_list)
    image_name_path = save_dir + image_name + '/'
    os.makedirs(image_name_path, exist_ok=True) 
    
    for img_path in image_path_list:
        
        # makeDir(save_dir + image_name) 
        file = os.path.basename(img_path)   # 拡張子ありファイル名を取得
        name, ext = os.path.splitext(file)  # 拡張子なしファイル名と拡張子を取得

        image = cv2.imread(img_path)
        image = cv2.resize(image, (resize_img))

        out_path = save_dir + image_name + '/' + name + '_resize' + ".jpg"

        cv2.imwrite(out_path, image) 
