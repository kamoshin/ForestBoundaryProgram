import os
import csv
import shutil
import pandas as pd
from sklearn.metrics import confusion_matrix

dataset_dir = './../../Data/trimed_img/fewdata_dataaugumentation/test/'
aug = 'rotate'
aug_dir = './../../Data/Report/few_rotate/'
wrong_img_dir = aug_dir + 'wrong_img/'
if not os.path.exists(wrong_img_dir):
    os.makedirs(wrong_img_dir)

label_dict = {0: 'japanese_cedar', 1: 'japanese_cypress', 2: 'other_tree', 3: 'not_tree'}

for key1 in label_dict:
    for key2 in label_dict:
        if not os.path.exists(wrong_img_dir+label_dict[key1]+'/'+label_dict[key2]):
            os.makedirs(wrong_img_dir+label_dict[key1]+'/'+label_dict[key2])    

orig_df = pd.read_csv('./../../Data/Report/few_data/wrong_img.csv', index_col=0)
aug_df = pd.read_csv(aug_dir+'wrong_img.csv', index_col=0)

image_path_list = orig_df.index.tolist()

t_list = []
y_list = []
for path in image_path_list:
    t = orig_df.at[path, 'true']
    orig_pred = orig_df.at[path, 'pred']
    aug_pred = aug_df.at[path, 'pred']

    if t == orig_pred and t != aug_pred:
        t_list.append(t)
        y_list.append(aug_pred)
        shutil.copy(dataset_dir+label_dict[t]+'/'+path, wrong_img_dir+label_dict[t]+'/'+label_dict[aug_pred]+'/'+path)

label_names = ['japanese_cedar', 'japanese_cypress', 'other_tree', 'not_tree']
confusion_matrix_df = pd.DataFrame(confusion_matrix(t_list, y_list), columns=label_names, index=label_names)
confusion_matrix_df.to_csv(aug_dir+'confusion_matrix_origtrue_augwrong.csv')