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
from utils.data import load_img, save_img, darken, gen_shadow,detect_shadow,batch_crop,calculate_psnr
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





continue_training=True
if ARGS.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=''
os.environ["CUDA_VISIBLE_DEVICES"]="3"
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
train_subdirs=["synthetic/with_corrn_reflection/train/others",
    "synthetic/with_syn_reflection/train/others",
    "real_world/train/others"]
val_dir="real_world/val/others"

train_img_names=np.array([train_img_name for subdir in train_subdirs 
    for train_img_name in sorted(glob(
        "%s/%s/*"%(data_root,subdir)))]).reshape((-1,5))
val_img_names=np.array(sorted(glob(
        "%s/%s/*"%(data_root,val_dir)))).reshape((-1,5))

train_num=train_img_names.shape[0]
val_num=val_img_names.shape[0]

train_ds=tf.data.Dataset.from_tensor_slices(train_img_names)
train_ds=train_ds.shuffle(train_num,reshuffle_each_iteration=False)
train_ds=train_ds.map(lambda x: [load_img(x[i],channels=3) for i in (0,1,2,4)],num_parallel_calls=4)
train_ds=train_ds.map(batch_crop,num_parallel_calls=4).prefetch(2)
val_ds=tf.data.Dataset.from_tensor_slices(val_img_names)
val_ds=val_ds.map(lambda x: [load_img(x[i],channels=3) for i in (0,1,2,4)])
val_ds=val_ds.map(batch_crop)
print(train_ds)

iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                           train_ds.output_shapes)
input_ambient,reflection,target,input_pureflash = iterator.get_next()

training_init_op = iterator.make_initializer(train_ds)
validation_init_op = iterator.make_initializer(val_ds)

with tf.variable_scope(tf.get_variable_scope()):
    # input_ambient=tf.placeholder(tf.float32,shape=[None,None,None,3])
    # input_pureflash=tf.placeholder(tf.float32,shape=[None,None,None,3])
    # # input_flash=tf.placeholder(tf.float32,shape=[None,None,None,3])
    # reflection=tf.placeholder(tf.float32,shape=[None,None,None,3])
    # target=tf.placeholder(tf.float32,shape=[None,None,None,3])

    # mask_shadow = tf.cast(tf.greater(input_pureflash, 0.02), tf.float32)
    # mask_highlight = tf.cast(tf.less(input_flash, 0.96), tf.float32)
    # mask_shadow_highlight = mask_shadow * mask_highlight


    gray_pureflash = 0.33 * (input_pureflash[...,0:1] + input_pureflash[...,1:2] + input_pureflash[...,2:3])
    bad_mask = detect_shadow(input_ambient, input_pureflash)
    reflection_layer = UNet_SE(tf.concat([input_ambient, gray_pureflash, (-bad_mask + 1)], axis=3), output_channel = 3, ext='Ref_')
    transmission_layer = UNet_SE(tf.concat([input_ambient, reflection_layer, (-bad_mask + 1)], axis=3), ext='Trans_')
    lossDict["percep_t"] = 0.1 * compute_percep_loss(target, transmission_layer, reuse=False)     
    lossDict["percep_r"] = 0.1 * compute_percep_loss(reflection, reflection_layer, reuse=True) 
    lossDict["total"] = lossDict["percep_t"] + lossDict["percep_r"]



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
    for id in range(val_num):
        fetch_list=[transmission_layer, reflection_layer, input_ambient, target, reflection,input_pureflash,target, lossDict]
        pred_image_t, pred_image_r, gt_input_ambient, gt_target, gt_reflection,tmp_pureflash,tmp_T, crtDict=sess.run(fetch_list)
        print("Epc: %3d, shape of outputs: "%epoch, pred_image_t.shape, pred_image_r.shape)
        tmp_psnr = calculate_psnr(pred_image_t[0], tmp_T[0])
        psnr.append(tmp_psnr)
        f.writelines('%s: %.6f\n'%(img_names[0], tmp_psnr))
        utils.save_concat_img(gt_input_ambient, gt_target, gt_reflection, tmp_pureflash, pred_image_t,pred_image_r, "./result/%s/%04d/val_%06d.jpg"%(model, epoch, id))
    mean_psnr = np.mean(psnr)
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
    for id in range(train_num):
        fetch_list=[opt, transmission_layer, reflection_layer, input_ambient, target, reflection,input_pureflash, lossDict]
        _, pred_image_t, pred_image_r, gt_input_ambient, gt_target, gt_reflection,tmp_pureflash, crtDict=sess.run(fetch_list)

        step += 1
        if step % 100 == 0:
            crtLoss_str = "   ".join(["{}: {:.3f}".format(key, value) for key, value in crtDict.items()])
            print("Epc:{:03d}-{:04d} | {} time:{:.3f}".format(epoch, id, crtLoss_str, time.time()-st) )
            st=time.time()    
            if step % 100 == 0:
                utils.save_concat_img(gt_input_ambient, gt_target, gt_reflection, tmp_pureflash, pred_image_t,pred_image_r, "./result/%s/%04d/train_%06d.jpg"%(model, epoch, id))

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

        img_names = sorted(glob("./data/synthetic/with_corrn_reflection/test/others" + '/*'))[:100]
        for id in range(len(img_names) // 5):
            tmp_pureflash, tmp_ambient, tmp_flash, tmp_T, tmp_R = load_paired_data(img_names, id)
            fetch_list=[transmission_layer, reflection_layer, input_ambient, target, reflection, lossDict]
            pred_image_t, pred_image_r, gt_input_ambient, gt_target, gt_reflection, crtDict=sess.run(fetch_list,
                feed_dict={input_ambient:tmp_ambient, reflection:tmp_R, target:tmp_T, input_pureflash:tmp_pureflash})
            print("Epc: %3d, shape of outputs: "%epoch, pred_image_t.shape, pred_image_r.shape)
            utils.save_concat_img(gt_input_ambient, gt_target, gt_reflection, tmp_pureflash, pred_image_t,pred_image_r, "./result/%s/%04d/val_real_%06d.jpg"%(model, epoch, id))

        img_names = sorted(glob("./data/synthetic/with_syn_reflection/test/others" + '/*'))[:100]
        for id in range(len(img_names) // 5):
            tmp_pureflash, tmp_ambient, tmp_flash, tmp_T, tmp_R = load_paired_data(img_names, id)
            h,w = tmp_T.shape[1:3]
            h = h // 32 * 32
            w = w // 32 * 32
            tmp_T, tmp_R, tmp_ambient, tmp_pureflash, tmp_flash = tmp_T[:,:h:2,:w:2,:], tmp_R[:,:h:2,:w:2,:], tmp_ambient[:,:h:2,:w:2,:], tmp_pureflash[:,:h:2,:w:2,:], tmp_flash[:,:h:2,:w:2,:]
            fetch_list=[transmission_layer, reflection_layer, input_ambient, target, reflection, lossDict]
            pred_image_t, pred_image_r, gt_input_ambient, gt_target, gt_reflection, crtDict=sess.run(fetch_list,
                feed_dict={input_ambient:tmp_ambient, reflection:tmp_R, target:tmp_T, input_pureflash:tmp_pureflash})
            print("Epc: %3d, shape of outputs: "%epoch, pred_image_t.shape, pred_image_r.shape)
            utils.save_concat_img(gt_input_ambient, gt_target, gt_reflection, tmp_pureflash, pred_image_t,pred_image_r, "./result/%s/%04d/val_fake_%06d.jpg"%(model, epoch, id))