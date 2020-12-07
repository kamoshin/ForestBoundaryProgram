import os
import shutil
from glob import glob
import random

def makeDir(dir_path, mode='remove'):
    if mode == 'remove':
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    elif mode == 'keep':
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    print('make directory '+dir_path)

dir_path = './../../../Data/trimed_img/unresize_randomsampling_train/'
image_name_list = ['japanese_cypress', 'japanese_cedar', 'other_tree', 'not_tree']
data_num = 3033//2

save_dir1 = './../../../Data/three_pronged/4/trimed_img/model1_data/'
save_dir2 = './../../../Data/three_pronged/4/trimed_img/model2_data/'
makeDir(save_dir2)
shutil.copytree(dir_path, save_dir1)

for image_name in image_name_list:
    image_dir = save_dir2 + image_name+'/'
    makeDir(image_dir)
    image_path_list = glob(save_dir1 + image_name +'/*.jpg')
    random.shuffle(image_path_list)
    for n in range(data_num):
        shutil.move(image_path_list[n], image_dir+os.path.basename(image_path_list[n]))
