import os.path as osp
import os

import tensorflow as tf
import numpy as np
import pandas as pd


def rgbg2rgb(rgbg):
    return tf.stack((rgbg[:,:,0],(rgbg[:,:,1]+rgbg[:,:,3])/2,rgbg[:,:,2]),axis=2)

def linref2srgb(linref):
    forward_matrix=tf.constant([[ 0.51215854,0.26953794,0.17924102],
                                        [ 0.10933821,0.85002241,0.04063938],
                                        [ 0.01164852 ,-0.43424423,1.24290822]])
    neutral_wb=tf.constant([0.59085778, 1. ,0.44278711])
    d50tosrgb=tf.constant([[3.1338561, -1.6168667, -0.4906146], 
                                    [-0.9787684, 1.9161415, 0.0334540], 
                                     [0.0719453, -0.2289914, 1.4052427]])
    linref=rgbg2rgb(linref)
    shape = tf.shape(linref)[0:2]
    height=shape[0]
    width=shape[1]
    linref = tf.math.minimum(linref,tf.reshape(neutral_wb,(1,1,3)))
    rgb_reshaped = tf.reshape(linref,(-1,3))
    camera2d50 = forward_matrix / tf.reshape(neutral_wb,(1,3))
    camera2srgb = tf.linalg.matmul(d50tosrgb,camera2d50)
    rgb_srgb = tf.linalg.matmul(rgb_reshaped, camera2srgb,transpose_b=True)
    orgshape_rgb_srgb = tf.reshape(rgb_srgb,(height, width,3))
    srgb =tf.pow(tf.clip_by_value(orgshape_rgb_srgb,0, 1),1/2.2)
    return srgb


def load_raw(image_path,channels=4,dtype=tf.dtypes.uint16,maximum=65535):
      return tf.cast(tf.image.decode_png(
      tf.io.read_file(image_path),channels=channels,dtype=dtype),dtype=tf.float32)/maximum
      


def crop_big(img,crop_size=576):
    n,h,w,c=tf.unstack(tf.shape(img))
    maximum=tf.maximum(h,w)
    minimum=tf.minimum(h,w)
    h2=h*22 // minimum *32
    w2=w*22 // minimum *32
    cropped=tf.cond(maximum >crop_size,
                    lambda :tf.image.random_crop(
                            tf.image.resize(img,(h2,w2)),
                                        (4,crop_size,crop_size,4)),
                        lambda: img )
    return cropped


def load_four_raw(img_paths):
    return tf.unstack(crop_big(
        tf.stack([load_raw(img_paths[i])for i in range(4)])))

def load_img(image_path,channels=0):
  return tf.image.convert_image_dtype(
      tf.image.decode_jpeg(
      tf.io.read_file(image_path),channels=channels),dtype=tf.float32)

def load_four_raw2rgb(img_paths):
    imgs =[load_img(img_paths[0],channels=3)]+[load_raw(img_paths[i])for i in range(1,4)]
    return tf.split(
        tf.image.resize(tf.image.random_crop(
        tf.concat(imgs,axis=2),(1280,1280,15)),(640,640)),(3,4,4,4),axis=2)

def load_raw_test(img_paths):
    raws=[load_raw(img_paths[i])for i in range(4)]
    shape = tf.shape(raws[0])[0:2]
    height=shape[0]//32 *32
    width=shape[1]//32 *32
    return tf.unstack(
        tf.image.random_crop(
        tf.stack([load_raw(img_paths[i])for i in range(4)]),(4,height,width,4)))