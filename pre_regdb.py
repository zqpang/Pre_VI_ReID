import os
import os.path as osp
import shutil
from PIL import Image
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Moco Training")
parser.add_argument("--dir", type=str, metavar='PATH', help="txt")
args = parser.parse_args()


file = open(args.dir, encoding = "utf-8")


old_root = '/home/zhiqi/dataset/RegDB'
new_root = '/home/zhiqi/dataset/regdb'

train_or_test = args.dir.strip().split("/")[-1].strip().split("_")[0]


def liner_trans(img, k, b=0):

    trans_list = [(np.float32(x) * k + b) for x in range(256)]

    trans_table = np.array(trans_list)

    trans_table[trans_table > 255] = 255
    trans_table[trans_table < 0] = 0
    trans_table = np.round(trans_table).astype(np.uint8)

    return cv2.LUT(img, trans_table)



dataset=[]  
label=[]

for line in file.readlines():    
    curLine=line.strip().split(" ")      
    dataset.append(curLine[0]) 
    label.append(curLine[-1])


for data in dataset:
    
    a = data.strip().split("/") 
    b = a[2].strip().split("_")
    if a[0]=='Visible' :
        camid = 1
    elif a[0]=='Thermal' :
        camid = 2

    old_dir = osp.join(old_root, data)
    
    img = Image.open(old_dir)
    if camid == 1:
        img = img.convert('L')
        img = img.convert('RGB')
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = liner_trans(img, -1, 255)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

           
    newfname = a[1]+'_'+str(camid)+'_'+str(b[3])+'.bmp'
    
    if train_or_test=='train' :
        save_root = osp.join(new_root, train_or_test)
    elif train_or_test=='test' :
        save_root = osp.join(new_root, train_or_test, a[0])
    
    if osp.exists(save_root) is False:
        os.makedirs(save_root)
        
    img.save(osp.join(save_root, newfname))
    



