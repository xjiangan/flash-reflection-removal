from glob import glob
import os.path as osp
import os

import numpy as np
import pandas as pd

from utils.data import write_split_list
seed = 2020#2019
np.random.seed(seed)


val_ratio=0.2

data_root="data"
img_dir=osp.join(data_root,"flashAmbient")
img_subdirs=["People_Photos","Objects_Photos",
         "Plants_Photos","Rooms_Photos",
         "Shelves_Photos","Toys_Photos"]
mask_dir=osp.join(data_root,"SynShadow")
mask_subdir="matte"

img_list=np.array([osp.join(subdir,img_name) for subdir in img_subdirs 
            for img_name in sorted(os.listdir(
                osp.join(img_dir,subdir))) if img_name.endswith(".png")]).reshape((-1,2))
mask_list= np.array([osp.join(mask_subdir,mask_name)
               for mask_name in os.listdir(osp.join(mask_dir,mask_subdir)) if mask_name.endswith(".png")])
write_split_list(img_list,[int(val_ratio*len(img_list))],
        img_dir,["val.csv","train.csv"],["ambient","flash"])
write_split_list(mask_list,[int(val_ratio*len(mask_list))],
        mask_dir,["val.csv","train.csv"],["mask"])

