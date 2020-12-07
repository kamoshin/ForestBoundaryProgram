import os
import csv
import shutil

orig_dir = './../../Data/trimed_img/test/'
wrong_img_dir = './../../Data/wrong_img/'
label_dict = {0: 'japanese_cedar', 1: 'japanese_cypress', 2: 'other_tree', 3: 'not_tree'}

for key1 in label_dict:
    for key2 in label_dict:
        if not os.path.exists(wrong_img_dir+label_dict[key1]+'/'+label_dict[key2]):
            os.makedirs(wrong_img_dir+label_dict[key1]+'/'+label_dict[key2])    

with open('./../../Data/Report/wrong_img.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        shutil.copy(orig_dir + label_dict[int(row[1])] + '/' + row[0], \
            wrong_img_dir + label_dict[int(row[1])] + '/' + label_dict[int(row[2])] + '/')