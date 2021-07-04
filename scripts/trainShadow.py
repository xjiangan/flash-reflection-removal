from __future__ import division
import os
import time
import math
import argparse
import subprocess
import random
from glob import glob

import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import numpy as np
import scipy.io
import scipy.stats as st
from scipy.misc import imread,imsave
import matplotlib.pyplot as plt

import utils.utils as utils
from utils.data import load_img, save_img, darken, gen_shadow
from model.network import UNet as UNet
from model.network import UNet_SE as UNet_SE
from loss.losses import compute_percep_loss

seed = 2020#2019
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--loss", default="lp", help="choose the loss type")
parser.add_argument("--is_test", default=0,type=int, help="choose the loss type")
parser.add_argument("--model", default="pre-trained",help="path to folder containing the model")
parser.add_argument("--debug", default=0, type=int, help="DEBUG or not")
parser.add_argument("--use_gpu", default=1, type=int, help="DEBUG or not")
parser.add_argument("--save_model_freq", default=10, type=int, help="frequency to save model")

ARGS = parser.parse_args()
DEBUG = ARGS.debug 
save_model_freq = ARGS.save_model_freq 
model=ARGS.model
is_test = ARGS.is_test
BATCH_SIZE=4

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64).clip(0,1)
    img2 = img2.astype(np.float64).clip(0,1)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


continue_training=True
if ARGS.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=''
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["OMP_NUM_THREADS"] = '4'
print(ARGS)



def detect_shadow(ambient, flashonly):
    intensity_ambient = tf.norm(ambient, axis=3, keepdims=True)
    intensity_flashonly = tf.norm(flashonly, axis=3, keepdims=True)
    ambient_ratio = intensity_ambient / tf.reduce_mean(intensity_ambient)
    flashonly_ratio = intensity_flashonly / tf.reduce_mean(intensity_flashonly)
    
    # Dark in PF but not dark in F
    pf_div_by_ambient = flashonly_ratio / (ambient_ratio+1e-5)
    shadow_mask = tf.cast(tf.less(pf_div_by_ambient, 0.8), tf.float32)
    
    # Cannot be too bright in flashonly
    dark_mask = tf.cast(tf.less(intensity_flashonly, 0.3), tf.float32)
    
    mask = dark_mask * shadow_mask
    return mask

def batch_crop(a,b,c,d):
    imgs=(a,b,c,d)
    batch=tf.stack(imgs)
    shape=tf.shape(batch)
    num=32
    den=32
    m=960
    out_shape=(4,tf.math.minimum(shape[1]//den * num,m) ,
                tf.math.minimum(shape[2]//den *num,m) ,3 )
    
    return [img[None,...] for img in tf.unstack(tf.image.random_crop(batch,out_shape))]

# set up the model and define the graph
lossDict= {}

data_root="data"
mask_dir="SynShadow/matte"
img_dir="flashAmbient"
img_subdirs=["People_Photos","Objects_Photos",
         "Plants_Photos","Rooms_Photos",
         "Shelves_Photos","Toys_Photos"]
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

train_ds=train_ds.map(lambda x:gen_shadow(x,mask_file_list),
            num_parallel_calls=4).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
val_ds=val_ds.map(lambda x:gen_shadow(x,mask_file_list)).batch(BATCH_SIZE)

print(train_ds)

iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                           train_ds.output_shapes)
img_with_shadow,shadow_mask,img_no_shadow,input_pureflash = iterator.get_next()

training_init_op = iterator.make_initializer(train_ds)
validation_init_op = iterator.make_initializer(val_ds)

with tf.variable_scope(tf.get_variable_scope()):
    # img_with_shadow=tf.placeholder(tf.float32,shape=[None,None,None,3])
    # input_pureflash=tf.placeholder(tf.float32,shape=[None,None,None,3])
    # # input_flash=tf.placeholder(tf.float32,shape=[None,None,None,3])
    # shadow_mask=tf.placeholder(tf.float32,shape=[None,None,None,3])
    # img_no_shadow=tf.placeholder(tf.float32,shape=[None,None,None,3])

    # mask_shadow = tf.cast(tf.greater(input_pureflash, 0.02), tf.float32)
    # mask_highlight = tf.cast(tf.less(input_flash, 0.96), tf.float32)
    # mask_shadow_highlight = mask_shadow * mask_highlight


    gray_pureflash = 0.33 * (input_pureflash[...,0:1] + input_pureflash[...,1:2] + input_pureflash[...,2:3])
    # bad_mask = detect_shadow(img_with_shadow, input_pureflash)
    shadow_mask_layer=shadow_mask
    # shadow_mask_layer = UNet_SE(tf.concat([img_with_shadow, gray_pureflash], axis=3), output_channel = 3, ext='Ref_')
    transmission_layer = UNet_SE(tf.concat([img_with_shadow, shadow_mask_layer], axis=3), ext='Trans_')
    lossDict["percep_t"] = 0.1 * compute_percep_loss(img_no_shadow, transmission_layer, reuse=False)
    lossDict["percep_r"] =tf.constant(0)     
    # lossDict["percep_r"] = 0.1 * compute_percep_loss(shadow_mask, shadow_mask_layer, reuse=True) 
    lossDict["total"] = lossDict["percep_t"]  # + lossDict["percep_r"]
    tf_psnr=tf.image.psnr(img_no_shadow,transmission_layer,1)



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

def load_paired_data(img_names, id):
    tmp_pureflash = imread(img_names[5 * id + 4])[None,...]/255.
    tmp_ambient = imread(img_names[5 * id])[None,...]/255.
    tmp_flash = imread(img_names[5 * id + 3])[None,...]/255.
    tmp_T = imread(img_names[5 * id + 2])[None,...]/255.
    tmp_R = imread(img_names[5 * id + 1])[None,...]/255.
    return tmp_pureflash, tmp_ambient, tmp_flash, tmp_T, tmp_R

def validation():
    img_names = sorted(glob("./data/real_world/val/others" + '/*'))
    psnr = []

    txt_path = "./result/%s/%04d/psnr_ssim.txt"%(model, epoch)
    f = open(txt_path,'w')

    sess.run(validation_init_op)
    id=0
    while True:
        try:
            fetch_list=[transmission_layer, shadow_mask_layer, img_with_shadow, img_no_shadow, shadow_mask,input_pureflash,img_no_shadow, lossDict]
            pred_image_t, pred_image_r, gt_img_with_shadow, gt_img_no_shadow, gt_shadow_mask,tmp_pureflash,tmp_T, crtDict=sess.run(fetch_list)
            tmp_psnr = calculate_psnr(pred_image_t[0], tmp_T[0])
            psnr.append(tmp_psnr)
            f.writelines('%s: %.6f\n'%(img_names[0], tmp_psnr))
            if id%100==0:
                print("Epc: %3d, shape of outputs: "%epoch, pred_image_t.shape, pred_image_r.shape)
                utils.save_concat_img(gt_img_with_shadow, gt_img_no_shadow, gt_shadow_mask, tmp_pureflash, pred_image_t,pred_image_r, "./result/%s/%04d/val_%06d.jpg"%(model, epoch, id))
            id+=1
        except tf.errors.OutOfRangeError:
            break

    mean_psnr = np.mean(psnr)
    print('%s: %.6f\n'%("average", mean_psnr))
    f.writelines('%s: %.6f\n'%("average", mean_psnr))
    f.close()
    return mean_psnr
best_psnr = 0


for epoch in range(1,maxepoch):
    print("Processing epoch %d"%epoch)
    # save model and images every epoch

    if os.path.isdir("./result/%s/%04d"%(model,epoch)):
        continue
    else:
        os.makedirs("./result/%s/%04d"%(model,epoch))

    if DEBUG:
        save_model_freq = 1

    sess.run(training_init_op)
    ept=time.time()

    st=time.time()
    id=0
    while True:
        try:
            fetch_list=[opt, transmission_layer, shadow_mask_layer, img_with_shadow, img_no_shadow, shadow_mask,input_pureflash, lossDict]
            _, pred_image_t, pred_image_r, gt_img_with_shadow, gt_img_no_shadow, gt_shadow_mask,tmp_pureflash, crtDict=sess.run(fetch_list)

            step += 1
            if step % 100 == 0:
                crtLoss_str = "   ".join(["{}: {:.3f}".format(key, value) for key, value in crtDict.items()])
                print("Epc:{:03d}-{:04d} | {} time:{:.3f}".format(epoch, id, crtLoss_str, time.time()-st) )
                st=time.time()    
                if step % 100 == 0:
                    utils.save_concat_img(gt_img_with_shadow, gt_img_no_shadow, gt_shadow_mask, tmp_pureflash, pred_image_t,pred_image_r, "./result/%s/%04d/train_%06d.jpg"%(model, epoch, id))
            id+=1
        except tf.errors.OutOfRangeError:
            break

    print("Epc:{:03d} |  time:{:.3f}".format(epoch,  time.time()-ept) )  
    mean_psnr = validation()
    if mean_psnr > best_psnr:
        best_psnr = mean_psnr
        print("mean: {:.2f}".format(mean_psnr))
        print("best: {:.2f}".format(best_psnr))
        saver.save(sess,"./result/%s/model.ckpt"%model)
        saver.save(sess,"./result/%s/%04d/model.ckpt"%(model,epoch-1))
    if (False) :#(is_test or (epoch % save_model_freq == 0 and epoch < 1000)):
        saver.save(sess,"./result/%s/model.ckpt"%model)
        saver.save(sess,"./result/%s/%04d/model.ckpt"%(model,epoch-1))

        img_names = sorted(glob("./data/synthetic/with_corrn_shadow_mask/test/others" + '/*'))[:100]
        for id in range(len(img_names) // 5):
            tmp_pureflash, tmp_ambient, tmp_flash, tmp_T, tmp_R = load_paired_data(img_names, id)
            fetch_list=[transmission_layer, shadow_mask_layer, img_with_shadow, img_no_shadow, shadow_mask, lossDict]
            pred_image_t, pred_image_r, gt_img_with_shadow, gt_img_no_shadow, gt_shadow_mask, crtDict=sess.run(fetch_list,
                feed_dict={img_with_shadow:tmp_ambient, shadow_mask:tmp_R, img_no_shadow:tmp_T, input_pureflash:tmp_pureflash})
            print("Epc: %3d, shape of outputs: "%epoch, pred_image_t.shape, pred_image_r.shape)
            utils.save_concat_img(gt_img_with_shadow, gt_img_no_shadow, gt_shadow_mask, tmp_pureflash, pred_image_t,pred_image_r, "./result/%s/%04d/val_real_%06d.jpg"%(model, epoch, id))

        img_names = sorted(glob("./data/synthetic/with_syn_shadow_mask/test/others" + '/*'))[:100]
        for id in range(len(img_names) // 5):
            tmp_pureflash, tmp_ambient, tmp_flash, tmp_T, tmp_R = load_paired_data(img_names, id)
            h,w = tmp_T.shape[1:3]
            h = h // 32 * 32
            w = w // 32 * 32
            tmp_T, tmp_R, tmp_ambient, tmp_pureflash, tmp_flash = tmp_T[:,:h:2,:w:2,:], tmp_R[:,:h:2,:w:2,:], tmp_ambient[:,:h:2,:w:2,:], tmp_pureflash[:,:h:2,:w:2,:], tmp_flash[:,:h:2,:w:2,:]
            fetch_list=[transmission_layer, shadow_mask_layer, img_with_shadow, img_no_shadow, shadow_mask, lossDict]
            pred_image_t, pred_image_r, gt_img_with_shadow, gt_img_no_shadow, gt_shadow_mask, crtDict=sess.run(fetch_list,
                feed_dict={img_with_shadow:tmp_ambient, shadow_mask:tmp_R, img_no_shadow:tmp_T, input_pureflash:tmp_pureflash})
            print("Epc: %3d, shape of outputs: "%epoch, pred_image_t.shape, pred_image_r.shape)
            utils.save_concat_img(gt_img_with_shadow, gt_img_no_shadow, gt_shadow_mask, tmp_pureflash, pred_image_t,pred_image_r, "./result/%s/%04d/val_fake_%06d.jpg"%(model, epoch, id))