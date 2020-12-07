import os
import sys
from PIL import Image
import numpy as np
import csv
from sklearn.metrics import classification_report
import pandas as pd

predict_img_path = ['./../../Data/Report/DenseNet/32+128/32+128/ushitsu_Dense32+128_32_boundarymap.png', './../../Data/Report/DenseNet/32+128/32+128/yuusenji_Dense32+128_32_boundarymap.png']
correct_img_path = ['./../../../OrthoData/test/colormap/ushitsu_boundarymap.png', './../../../OrthoData/test/colormap/yuusenji_boundarymap.png']

label_names = ['japanese_cedar', 'japanese_cypress', 'other_tree', 'not_tree', 'not_image', 'boundary']

weight = '32+128'
report_dir = './../../Data/Report/DenseNet/'
formula_dir = ['32+128']

check_size = 32
pred_lab_all = []
corr_lab_all = []
for img_path_n in range(len(predict_img_path)):
    pred_lab = []
    corr_lab = []
    print('load image')
    print(predict_img_path[img_path_n])
    # pred_colormap = Image.open(report_dir+formula+'/'+weight+'/'+predict_img_path[img_path_n]+'_Dense'+weight+'_pred_colormap.png')
    pred_colormap = Image.open(predict_img_path[img_path_n])
    corr_colormap = Image.open(correct_img_path[img_path_n])

    pred_colormap = pred_colormap.convert('RGB')
    corr_colormap = corr_colormap.convert('RGB')

    size = pred_colormap.size
    width = size[0]
    height = size[1]
    print(width, height)

    for y in range(0, height, check_size):
        if y+check_size > height:
            break
        sys.stdout.write('\r pred colormap y:%d' % y)
        sys.stdout.flush()
        for x in range(0, width, check_size):
            if x+check_size > width:
                break
            #predicted colormap
            pred_color_count = [0,0,0,0,0,0]
            corr_color_count = [0,0,0,0,0,0]
            for r_y in range(y, y+check_size):
                for r_x in range(x, x+check_size):
                    r,g,b = pred_colormap.getpixel((r_x,r_y))
                    # print(r_pred_colormap[r_y][r_x])
                    if r==255 and g==0 and b==0:
                        pred_color_count[0] += 1
                    elif r==0 and g==128 and b==0:
                        pred_color_count[1] += 1
                    elif r==255 and g==255 and b==0:
                        pred_color_count[2] += 1
                    elif r==128 and g==128 and b==128:
                        pred_color_count[3] += 1
                    elif r==0 and g==0 and b==0:
                        pred_color_count[4] += 1
                    elif r==160 and g==32 and b==255:
                        pred_color_count[5] += 1
                    else:
                        print('pred colormap error')
                        exit()

                    #corr colormap               
                    r,g,b = corr_colormap.getpixel((r_x,r_y))
                    if r==255 and g==0 and b==0:
                        corr_color_count[0] += 1
                    elif r==0 and g==128 and b==0:
                        corr_color_count[1] += 1
                    elif r==255 and g==255 and b==0:
                        corr_color_count[2] += 1
                    elif r==128 and g==128 and b==128:
                        corr_color_count[3] += 1
                    elif r==0 and g==0 and b==0:
                        corr_color_count[4] += 1
                    elif r==160 and g==32 and b==255:
                        corr_color_count[5] += 1
                    else:
                        print('pred colormap error')
                        exit() 
            pred_lab_all.append(np.argmax(np.array(pred_color_count)))
            pred_lab.append(np.argmax(np.array(pred_color_count)))
            corr_lab_all.append(np.argmax(np.array(corr_color_count)))
            corr_lab.append(np.argmax(np.array(corr_color_count))) 

    report_df = pd.DataFrame(classification_report(corr_lab, pred_lab, labels = range(len(label_names)), target_names=label_names, output_dict=True)).transpose()
    print(report_df)
    # save_path = report_dir+formula+'/'+weight+'/'+'comp_colormap_'+predict_img_path[img_path_n]+'.csv'
    # print(save_path)
    # report_df.to_csv(save_path)


report_df = pd.DataFrame(classification_report(corr_lab_all, pred_lab_all, labels = range(len(label_names)), target_names=label_names, output_dict=True)).transpose()
print(report_df)
# save_path = report_dir+formula+'/'+weight+'/'+'comp_colormap_all.csv'
# print(save_path)
# report_df.to_csv(save_path)