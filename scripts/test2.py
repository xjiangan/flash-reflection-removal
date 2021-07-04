from __future__ import division
import os
import time
import argparse
import subprocess
import random
from glob import glob

import tensorflow as tf
import numpy as np

import utils.utils as utils
from utils.data import load_imgs, save_img, darken, gen_shadow,detect_shadow,batch_crop
from utils.data import concat_img, encode_jpeg
from model.network import UNet as UNet
from model.network import UNet_SE as UNet_SE
from loss.losses import compute_percep_loss


seed = 2020#2019
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="full_global_lp",help="path to folder containing the model")
parser.add_argument("--mask_dir", default="./data/demo",help="path to folder containing the model")
parser.add_argument("--out_dir", default="./data/demo",help="path to folder containing the model")
parser.add_argument("--testset", default="./data/demo",help="path to folder containing the model")
ARGS = parser.parse_args()
model=ARGS.model
testset=ARGS.testset
mask_dir=ARGS.mask_dir
out_dir=ARGS.out_dir

continue_training=True
os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
os.environ["OMP_NUM_THREADS"] = '4'
print(ARGS)
BATCH_SIZE=1

# exit()

# set up the model and define the graph
lossDict= {}

mask_file_list=np.array(sorted(
    glob("%s/*"%mask_dir))).reshape((-1,1))

img_names=np.concatenate((np.array([img_name for img_name in sorted(glob(
        "%s/*"%testset))]).reshape((-1,3)),mask_file_list),axis=1)
num_img=img_names.shape[0]

val_ds=tf.data.Dataset.from_tensor_slices(img_names)

val_ds=val_ds.map(lambda x:load_imgs(x,4)).batch(BATCH_SIZE)

print(val_ds)

iterator = tf.data.Iterator.from_structure(val_ds.output_types,
                                           val_ds.output_shapes)
img_with_shadow,input_pureflash,img_no_shadow,shadow_mask = iterator.get_next()

validation_init_op = iterator.make_initializer(val_ds)

with tf.variable_scope(tf.get_variable_scope()):

    gray_pureflash = 0.33 * (input_pureflash[...,0:1] + input_pureflash[...,1:2] + input_pureflash[...,2:3])
    # bad_mask = detect_shadow(img_with_shadow, input_pureflash)
    shadow_mask_layer = UNet_SE(tf.concat([img_with_shadow, gray_pureflash], axis=3), output_channel = 3, ext='Ref_')
    no_shadow_layer = UNet_SE(tf.concat([img_with_shadow, shadow_mask_layer], axis=3), ext='Trans_')
    lossDict["percep_t"] = 0.1 * compute_percep_loss(img_no_shadow, no_shadow_layer, reuse=False)     
    lossDict["percep_r"] = 0.1 * compute_percep_loss(shadow_mask, shadow_mask_layer, reuse=True) 
    lossDict["total"] = lossDict["percep_t"] + lossDict["percep_r"]
    tf_psnr=tf.math.reduce_mean(tf.image.psnr(tf.clip_by_value(img_no_shadow,0,1),
                        tf.clip_by_value(no_shadow_layer,0,1),1.0))
    encoded_concat=encode_jpeg(
        concat_img((img_with_shadow[0],no_shadow_layer[0],img_no_shadow[0],
            input_pureflash[0],shadow_mask_layer[0],shadow_mask[0])))




train_vars = tf.trainable_variables()

R_vars = [var for var in train_vars if 'Ref_' in var.name]
T_vars = [var for var in train_vars if 'Trans_' in var.name]
all_vars=[var for var in train_vars if 'g_' in var.name]

for var in R_vars: 	print(var)
for var in T_vars:	print(var)
opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(lossDict["total"],var_list=all_vars)


for var in tf.trainable_variables():
    print("Listing trainable variables ... ")
    print(var)



######### Session #########
saver = tf.train.Saver(max_to_keep=20)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# writer = tf.summary.FileWriter("./result/"+model)
# writer.add_graph(sess.graph)
sess.run(tf.global_variables_initializer())
var_restore = [v for v in tf.trainable_variables()]
saver_restore = tf.train.Saver(var_restore)
ckpt = tf.train.get_checkpoint_state('./result/'+model)
######### Session #########


print("[i] contain checkpoint: ", ckpt)
if ckpt and continue_training:
    saver_restore=tf.train.Saver([var for var in tf.trainable_variables()])
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restore.restore(sess,ckpt.model_checkpoint_path)


maxepoch=151
step = 0
val_save_freq=1
train_save_freq=200

best_psnr = 0


epoch_dir=out_dir
if not os.path.exists(epoch_dir):
    os.makedirs(epoch_dir)

######### Validation #########
psnr = []
f = open("%s/psnr_ssim.txt"%epoch_dir,'w')
sess.run(validation_init_op)
batch_id=0
while True:
    try:
        fetch_dict={"psnr":tf_psnr}
        if batch_id%val_save_freq==0:
            fetch_dict["concat"]=encoded_concat
        tmp=sess.run(fetch_dict)
        psnr.append(tmp["psnr"])
        f.writelines('%4d: %.6f\n'%(batch_id, tmp["psnr"]))
        if batch_id%val_save_freq==0:
            with open("%s/val_%06d.jpg"%(epoch_dir, batch_id),"wb") as img_f:
                img_f.write(tmp["concat"])
        batch_id+=1
    except tf.errors.OutOfRangeError:
        break

mean_psnr = np.mean(psnr)
f.writelines('%s: %.6f\n'%("average", mean_psnr))
f.close()
######### Validation #########



