from __future__ import division
from glob import glob
import os.path as osp
import os
import random
import subprocess
import argparse

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data import load_img , darken, gen_istd
os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
seed = 2020#2019
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="data")
parser.add_argument("--out", default="dbg")

ARGS = parser.parse_args()


data_root=ARGS.data_root
output_dir=osp.join(data_root,ARGS.out)

mask_dir=osp.join(data_root,"SynShadow")
mask_subdir="matte"
img_dir=osp.join(data_root,"flashAmbient")
img_subdirs=["People_Photos","Objects_Photos",
         "Plants_Photos","Rooms_Photos",
         "Shelves_Photos","Toys_Photos"]


train_out=osp.join(output_dir,"train")
test_out=osp.join(output_dir,"test")
for p in(train_out,test_out):
    if not osp.exists(p):
        os.makedirs(p)

gen_istd(img_dir,mask_dir,"train.csv",train_out,"train")
gen_istd(img_dir,mask_dir,"val.csv",test_out,"test")