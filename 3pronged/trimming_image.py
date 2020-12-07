"""
Created on Fri Nov 16 21:25:18 2018

@author: kawasaki
"""
import os
from cv2 import cv2
import numpy as np

import matplotlib.pyplot as plt
#---Matlab function---
def showImagePLT(im):
    plt.imshow(im)
    plt.show()

def makeDir(dir_path, mode='keep'):
    if mode == 'remove':
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
            print('remove directory '+dir_path)
        os.makedirs(dir_path)
        print('make directory '+dir_path)
    elif mode == 'keep':
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print('make directory '+dir_path)

# Target RGB color of map_image    
def selectColor(target_name):
    select_color = (0,0,0)    
    if target_name == "japanese_cedar": # スギ
        select_color = (255,0,0)
    elif target_name == "japanese_cypress": # ヒノキ
        select_color = (0,128,0)
    elif target_name == "other_tree": # 他樹種       
        select_color = (255,255,0)
    elif target_name == "not_tree":    
        select_color = (128,128,128) # 樹種以外
    else :
        print("error select color")
    
    return select_color

sampling_image_size = (256, 256)

orig_dir = './../../../../OrthoData/train/'

save_dir1 = './../../../Data/three_pronged/4/trimed_img/256/model1_data/'
save_dir2 = './../../../Data/three_pronged/4/trimed_img/256/model2_data/'
makeDir(save_dir1)

def main():
    photo_name_list1 = ['bamba','takemata','yoshikuraB','kurikara','urii']
    photo_name_list2 = ['rokuro','bodai','nashitani','oritani','togi','matii']

    for photo_name in photo_name_list1:
        # read image
        map_image = cv2.imread(orig_dir+"colormap/"+photo_name+"_colormap.png")
        aerial_image = cv2.imread(orig_dir+"aerial/"+photo_name+".jpg")
        # resize image
        #map_image = cv2.resize(map_image,image_size)
        #aerial_image = cv2.resize(aerial_image,image_size)
        # convert color ( refer : https://note.nkmk.me/python-opencv-bgr-rgb-cvtcolor/ )
        map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
        aerial_image = cv2.cvtColor(aerial_image, cv2.COLOR_BGR2RGB)
                
        # show image    
        #showImagePLT(map_image)
        #showImagePLT(aerial_image)
        target_name_list = ['japanese_cedar', 'japanese_cypress', 'other_tree', 'not_tree']
        #target_name_list = ['japanese_cypress']
        for target_name in target_name_list:
            # select target            
            target_color = selectColor(target_name)
            # check isdir
            makeDir(save_dir1+target_name)                
            
            height, width, channels = aerial_image.shape[:3]
            slide = 64
            
            save_image_num = 0
            w = sampling_image_size[0]
            h = sampling_image_size[1]
            # for loop, image size ( refer : https://techacademy.jp/magazine/15828 )
            for x in range(0,width,slide):
                for y in range(0,height,slide):
                    # trimming ( refer : https://algorithm.joho.info/programming/python/opencv-roi-py/ )
                    trimming_map_image = map_image[y:y+h, x:x+w]
                    
                    # extraction color
                    # convert target color is 255, others 0
                    bin_image = cv2.inRange(trimming_map_image, target_color, target_color)
                    
                    black_in_image = cv2.inRange(trimming_map_image, (0,0,0), (0,0,0)) #画像の黒い部分(背景)が含まれていてもデータとして取られてしまうのを防ぐための変数
                    
                    # count for array value ( refer : https://note.nkmk.me/python-numpy-count/ )
                    # save image, when target color sum equal all pixel
                    # - all pixel = (w*h), target color sum = np.sum(bin_image == 255)         
                    if np.sum(bin_image == 255) == w*h and np.sum(black_in_image == 255) == 0:
                        # trimming
                        trimming_color_image = aerial_image[y:y+h, x:x+w]
                        
                        # save color image
                        point = "_x"+str(x)+"y"+str(y)
                        save_name = save_dir1+target_name+"/"+photo_name+"_"+str(save_image_num)+point+".jpg"
                        save_image = cv2.cvtColor(trimming_color_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_name,save_image)
                        # save map image
                        """                
                        save_name = save_dir+target_name+"/map_"+str(save_image_num)+point+".png"
                        save_image = cv2.cvtColor(trimming_map_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_name,save_image)
                        """
                        save_image_num += 1
                        
                        
            print("{} image num :{}".format(target_name, save_image_num))

if __name__ == '__main__':
    main()