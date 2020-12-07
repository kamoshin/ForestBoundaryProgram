import os
import shutil
from cv2 import cv2
import numpy as np

def makeDir(dir_path, mode='keep'):
    if mode == 'remove':
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    elif mode == 'keep':
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

target_img_dir = './../../../Data/three_pronged/ensemble/pseudo_label_img/'
# target_img_list = os.listdir(target_img_dir)
target_img_list = ['ushitsu', 'yuusenji']

save_dir = './../../../Data/trimed_img/fewdata_dataaugumentation/samplingtraindata1000_add_pseudo_label/'
makeDir(save_dir)

# contains_pseudo_data = shutil.copytree('./../../../Data/trimed_img/train/',\
#                                        './../../../Data/trimed_img/train_and_pseudo_data/')

# label_names = ['japanese_cedar', 'japanese_cypress', 'other_tree', 'not_tree']
target_labels = ['japanese_cypress', 'other_tree', 'not_tree']

for target_img in target_img_list:
    print(target_img)
    for label in target_labels:
        pseudo_img_list = os.listdir(target_img_dir+target_img+'/'+label+'/')
        cc = 0
        for pseudo_img_path in pseudo_img_list:
            img = cv2.imread(target_img_dir+'/'+target_img+'/'+label+'/'+pseudo_img_path)
            white_in_img = cv2.inRange(img, (255,255,255), (255,255,255))
            black_in_img = cv2.inRange(img, (0, 0, 0), (0, 0, 0))
            height, width, channels = img.shape[:3]
            if np.sum(white_in_img == 255) > 0:
                continue
            if np.sum(black_in_img == 255) > 0:
                continue
            if height*width < 256*256:
                continue
            shutil.copy(target_img_dir+'/'+target_img+'/'+label+'/'+pseudo_img_path, \
            save_dir+label+'/')
            cc += 1
        print(cc)