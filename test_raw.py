from __future__ import division
import os
import time
import argparse
import subprocess
import random
import os.path as osp
from glob import glob

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

import utils.utils as utils
from utils.data import load_img, save_img, darken, gen_shadow,detect_shadow,batch_crop
from utils.data import concat_img, encode_jpeg,load_four
from utils.raw import load_four_raw,linref2srgb,rgbg2rgb,load_raw_test
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
parser.add_argument("--gpu", default=3, type=int, help="DEBUG or not")
parser.add_argument("--save_model_freq", default=10, type=int, help="frequency to save model")

ARGS = parser.parse_args()
DEBUG = ARGS.debug 
save_model_freq = ARGS.save_model_freq 
model=ARGS.model
is_test = ARGS.is_test
BATCH_SIZE=1

NOFLASH=False

continue_training=True
if ARGS.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=''

os.environ["OMP_NUM_THREADS"] = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
print(ARGS)



# set up the model and define the graph
lossDict= {}

data_root="data/docshadow"
test_dirs=["bio2"]

test_dfs=[]
for subdir in test_dirs:
    df=pd.read_csv(osp.join(data_root,subdir,'trip.csv'))
    df["f"]=df["f"].map(lambda x: osp.join('rawc','derived',x))
    df["m"]=df["ab"].map(lambda x:osp.join('rawc','derived',x))
    df[["gt","ab"]]=df[["gt","ab"]].applymap(lambda x: osp.join('rawc','origin',x))
    df=df.applymap(lambda x:osp.join(data_root,subdir,x+'.png'))
    test_dfs.append(df)
test_df=pd.concat(test_dfs)

test_arr=test_df.to_numpy()
test_size=len(test_arr)
test_ds=tf.data.Dataset.from_tensor_slices(test_arr)
test_ds=test_ds.shuffle(test_size,reshuffle_each_iteration=False)

test_ds=test_ds.map(lambda x:load_raw_test(x)).batch(1)

print(test_ds)
test_size=len(test_arr)
print(len(test_arr))
iterator = tf.data.Iterator.from_structure(test_ds.output_types,
                                           test_ds.output_shapes)
img_no_shadow,img_with_shadow,input_pureflash,shadow_mask = iterator.get_next()

test_init_op = iterator.make_initializer(test_ds)


with tf.variable_scope(tf.get_variable_scope()):

    gray_pureflash = 0.25 * (input_pureflash[...,0:1] + input_pureflash[...,1:2] + input_pureflash[...,2:3]+input_pureflash[...,3:4])
    # bad_mask = detect_shadow(img_with_shadow, input_pureflash)

    if NOFLASH:
        shadow_mask_layer = UNet_SE(tf.concat([img_with_shadow], axis=3), output_channel = 4, ext='Ref_')
    else:
        shadow_mask_layer = UNet_SE(tf.concat([img_with_shadow, gray_pureflash], axis=3), output_channel = 4, ext='Ref_')
                        # tf.math.sigmoid()
    no_shadow_layer = UNet_SE(tf.concat([img_with_shadow, shadow_mask_layer], axis=3),output_channel = 4, ext='Trans_')
    # lossDict["percep_t"] = 0.1 * compute_percep_loss(img_no_shadow, no_shadow_layer, reuse=False)    
    lossDict["percep_t"]=0.1* tf.reduce_mean(tf.abs(img_no_shadow- no_shadow_layer))
    # lossDict["percep_r"] = 0.1 * compute_percep_loss(shadow_mask, shadow_mask_layer, reuse=True) 
    lossDict["percep_r"]=0.1* tf.reduce_mean(tf.abs(shadow_mask-shadow_mask_layer))
    lossDict["total"] = lossDict["percep_t"] + lossDict["percep_r"]
    tf_psnr=tf.math.reduce_mean(tf.image.psnr(tf.clip_by_value(img_no_shadow,0,1),
                        tf.clip_by_value(no_shadow_layer,0,1),1.0))
    encoded_concat=encode_jpeg(
        concat_img((linref2srgb(img_with_shadow[0]),linref2srgb(no_shadow_layer[0]),linref2srgb(img_no_shadow[0]),
            linref2srgb(input_pureflash[0]),rgbg2rgb(shadow_mask_layer[0]), rgbg2rgb(shadow_mask[0]))))

######### Session #########
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

result_dir="./result/%s/test"%(model)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


######### Validation #########
psnr = []
f = open("%s/psnr_ssim.txt"%result_dir,'w')
sess.run(test_init_op)
batch_id=0
for _ in tqdm(range(test_size)):
    try:
        fetch_dict={"psnr":tf_psnr,'concat':encoded_concat}
        tmp=sess.run(fetch_dict)
        psnr.append(tmp["psnr"])
        f.writelines('%4d: %.6f\n'%(batch_id, tmp["psnr"]))
        with open("%s/test_%06d.jpg"%(result_dir, batch_id),"wb") as img_f:
            img_f.write(tmp["concat"])
        batch_id+=1
    except tf.errors.OutOfRangeError:
        break

mean_psnr = np.mean(psnr)
f.writelines('%s: %.6f\n'%("average", mean_psnr))
f.close()
######### Validation #########



