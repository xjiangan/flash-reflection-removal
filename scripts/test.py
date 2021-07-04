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
parser.add_argument("--testset", default="./data/demo",help="path to folder containing the model")
ARGS = parser.parse_args()
model=ARGS.model
testset=ARGS.testset
mask_dir=ARGS.mask_dir

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
        "%s/*"%testset))]).reshape((-1,2)),mask_file_list),axis=1)
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



######### Session #########
saver = tf.train.Saver(max_to_keep=20)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
var_restore = [v for v in tf.trainable_variables()]
saver_restore = tf.train.Saver(var_restore)
ckpt = tf.train.get_checkpoint_state('./ckpt/'+model)
######### Session #########


print("[i] contain checkpoint: ", ckpt)
if ckpt and continue_training:
    saver_restore=tf.train.Saver([var for var in tf.trainable_variables()])
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restore.restore(sess,ckpt.model_checkpoint_path)


data_dir = "{}/others".format(ARGS.testset)
data_names = sorted(glob(data_dir+"/*ambient.jpg"))

def crop_shape(tmp_all, size=32):
    h,w = tmp_all.shape[1:3]
    h = h // size * size
    w = w // size * size
    return h, w

num_test = len(data_names)
print(num_test)
for epoch in range(9999, 10000):
    print("Processing epoch %d"%epoch, "./ckpt/%s/%s"%(model,data_dir.split("/")[-2]))
    # save model and images every epoch
    save_dir = "./ckpt/%s/%s"%(model,data_dir.split("/")[-2])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    print("output path: {}".format(save_dir))
    all_loss_test=np.zeros(num_test, dtype=float)
    metrics = {"T_ssim":0,"T_psnr":0,"R_ssim":0, "R_psnr":0}
    fetch_list=[transmission_layer, reflection_layer, input_ambient, input_flash, input_pureflash, bad_mask]
    for id in tqdm(range(num_test)):
        st=time.time()
        tmp_pureflash = imread(data_names[id].replace("ambient.jpg", "pureflash.jpg"))[None,...] / 255.
        tmp_ambient = imread(data_names[id])[None,...] / 255.
        tmp_flash = imread(data_names[id].replace("ambient.jpg", "flash.jpg"))[None,...] / 255.
        h,w = crop_shape(tmp_ambient, size=32)
        tmp_ambient, tmp_pureflash, tmp_flash = tmp_ambient[:,:h,:w,:], tmp_pureflash[:,:h,:w,:], tmp_flash[:,:h,:w,:]
        pred_image_t, pred_image_r, in_ambient, in_flash, in_pureflash, pred_mask = sess.run(fetch_list,
            feed_dict={input_ambient:tmp_ambient, input_pureflash:tmp_pureflash, input_flash: tmp_flash})
        # print("Epc: %3d, shape of outputs: "%epoch, pred_image_t.shape, pred_image_r.shape)
        save_path = "{}/{}".format(save_dir, data_names[id].split("/")[-1])
        imsave(save_path.replace("ambient.jpg", "_0_input_ambient.png"), np.uint8(tmp_ambient[0].clip(0,1) * 255.))
        imsave(save_path.replace("ambient.jpg", "_1_pred_transmission.png"), np.uint8(pred_image_t[0].clip(0,1) * 255.))
        # imsave(save_path.replace("ambient.jpg", "_2_pred_refletion.png"), np.uint8(pred_image_r[0].clip(0,1) * 255.))
        imsave(save_path.replace("ambient.jpg", "_3_input_flash.png"), np.uint8(tmp_flash[0].clip(0,1) * 255.))
        imsave(save_path.replace("ambient.jpg", "_4_input_pureflash.png"), np.uint8(tmp_pureflash[0].clip(0,1) * 255.))
        # imsave(save_path.replace("ambient.jpg", "_5_mask.png"), np.uint8(pred_mask[0,...,0].clip(0,1) * 255.))

