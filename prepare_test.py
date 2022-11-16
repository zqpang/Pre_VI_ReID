import cv2
import glob
import os.path as osp
import os
from skimage import exposure

rgb_path = r"D:\dataset\reid\RGB-IR\sysu_new\test\rgb"
target_path = r"D:\dataset\reid\RGB-IR\sysu_new\test\gray"


img_paths = glob.glob(osp.join(rgb_path, '*.jpg'))
list_dirs = os.walk(rgb_path)

for _, _, files in list_dirs:
    for file in files:
        img = cv2.imread(osp.join(rgb_path, file))
        #img = img[:, :, 2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = exposure.adjust_gamma(img, 0.5)
        cv2.imwrite(osp.join(target_path, file), img)

