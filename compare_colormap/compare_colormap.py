import os
import sys
from PIL import Image
import numpy as np
import csv
from sklearn.metrics import classification_report
import pandas as pd

predict_img_path = ['ushitsu']
correct_img_path = ['./../../../OrthoData/test/colormap/ushitsu_colormap.png']

label_names = ['japanese_cedar', 'japanese_cypress', 'other_tree', 'not_tree', 'not_image']

weight = '32'
report_dir = './../../Data/Report/DenseNet/'
formula_dir = ['32']

for formula in formula_dir:
    print(formula)
    weight_dir = os.listdir(report_dir+formula+'/')
    for weight in weight_dir:
        print(weight)
        pred_lab_all = []
        corr_lab_all = []
        pred_colormap_path = os.listdir(report_dir+formula+'/'+weight+'/')
        for img_path_n in range(len(predict_img_path)):
            pred_lab = []
            corr_lab = []
            print('load image')
            img1 = Image.open(report_dir+formula+'/'+weight+'/'+predict_img_path[img_path_n]+'_Dense'+weight+'_pred_colormap.png')
            img2 = Image.open(correct_img_path[img_path_n])
            
            print('convert RGB')
            img1 = img1.convert('RGB')
            img2 = img2.convert('RGB')

            size = img1.size

            pixel_num = 0
            for x in range(size[0]):
                for y in range(size[1]):
                    pixel_num += 1
                    sys.stdout.write('\r pixel_num:%d' % pixel_num)
                    sys.stdout.flush()

                    r,g,b = img1.getpixel((x,y))
                    if r==255 and g==0 and b==0:
                        pred_lab_all.append(0)
                        pred_lab.append(0)
                    elif r==0 and g==128 and b==0:
                        pred_lab_all.append(1)
                        pred_lab.append(1)
                    elif r==255 and g==255 and b==0:
                        pred_lab_all.append(2)
                        pred_lab.append(2)
                    elif r==128 and g==128 and b==128:
                        pred_lab_all.append(3)
                        pred_lab.append(3)
                    elif r==0 and g==0 and b==0:
                        pred_lab_all.append(4)
                        pred_lab.append(4)
                    else:
                        print('img1 error')
                        exit()
                        
                    r,g,b = img2.getpixel((x,y))
                    if r==255 and g==0 and b==0:
                        corr_lab_all.append(0)
                        corr_lab.append(0)
                    elif r==0 and g==128 and b==0:
                        corr_lab_all.append(1)
                        corr_lab.append(1)
                    elif r==255 and g==255 and b==0:
                        corr_lab_all.append(2)
                        corr_lab.append(2)
                    elif r==128 and g==128 and b==128:
                        corr_lab_all.append(3)
                        corr_lab.append(3)
                    elif r==0 and g==0 and b==0:
                        corr_lab_all.append(4)
                        corr_lab.append(4)
                    else:
                        print('img2 error')
                        exit()

            report_df = pd.DataFrame(classification_report(corr_lab, pred_lab, labels = range(len(label_names)), target_names=label_names, output_dict=True)).transpose()
            print(report_df)
            save_path = report_dir+formula+'/'+weight+'/'+'comp_colormap_'+predict_img_path[img_path_n]+'.csv'
            print(save_path)
            report_df.to_csv(save_path)


        report_df = pd.DataFrame(classification_report(corr_lab_all, pred_lab_all, labels = range(len(label_names)), target_names=label_names, output_dict=True)).transpose()
        print(report_df)
        save_path = report_dir+formula+'/'+weight+'/'+'comp_colormap_all.csv'
        print(save_path)
        report_df.to_csv(save_path)