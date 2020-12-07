import os
import shutil
from glob import glob
import random

def makeDir(dir_path, mode='remove'):
    if mode == 'remove':
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    elif mode == 'keep':
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

# dir_path = './../../Data/images/'
dir_path = './../../../Data/three_pronged/4/trimed_img/256/model1_data/'
#dir_path = './../../../Data/three_pronged/4/trimed_img/256/model2_data/'
image_name_list = ['japanese_cypress', 'japanese_cedar', 'other_tree', 'not_tree']
# image_name_list = ['japanese_cedar']
#image_name_list = ['other_tree', 'not_tree']
# data_num = 4916
#data_num = 1637
data_num = 1396

# save_dir = './../../Data/randomsamplingdata/'
save_dir = './../../../Data/three_pronged/4/trimed_img/random_sampling/model1_data/'
#save_dir = './../../../Data/three_pronged/4/trimed_img/random_sampling/model2_data/'
makeDir(save_dir)

for image_name in image_name_list:
    image_dir = save_dir+image_name+'/'
    makeDir(image_dir)
    image_path_list = glob(dir_path + image_name +'/*.jpg')
    random.shuffle(image_path_list)
    for n in range(data_num):
        shutil.copyfile(image_path_list[n], image_dir+os.path.basename(image_path_list[n]))