from __future__ import division
from glob import glob

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.data import load_img, save_img, darken, gen_shadow

data_root="data"
mask_dir="SynShadow/matte"
img_dir="flashAmbient"
img_subdirs=["People_Photos","Objects_Photos",
         "Plants_Photos","Rooms_Photos",
         "Shelves_Photos","Toys_Photos"]
output_dir="Shadow"

mask_file_list=tf.constant(sorted(
    glob("%s/%s/*"%(data_root,mask_dir))))

img_names=np.array([img_name for subdir in img_subdirs 
    for img_name in sorted(glob(
        "%s/%s/%s/*"%(data_root,img_dir,subdir)))]).reshape((-1,2))
num_img=img_names.shape[0]

ds=tf.data.Dataset.from_tensor_slices(img_names)
ds=ds.shuffle(num_img,reshuffle_each_iteration=False)

val_size=int(num_img*0.2)
train_size=num_img-val_size
train_ds =ds.skip(val_size)
val_ds =ds.take(val_size)

train_ds=train_ds.map(lambda x:gen_shadow(x,mask_file_list))
val_ds=val_ds.map(lambda x:gen_shadow(x,mask_file_list))


n=100
i=0
class_names=["noshad","flash","shadow","mask"]
iterator=train_ds.take(n).make_one_shot_iterator()
next_imgs=iterator.get_next()
with tf.Session() as sess:
    for _ in range(n):
        imgs=sess.run(next_imgs)
        i+=1
        for j, cls in enumerate(class_names):
            sess.run(save_img("%s/%04d_%s.jpg"%(output_dir,i,cls),imgs[j]))

