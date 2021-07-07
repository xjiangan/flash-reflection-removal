import os.path as osp
import os

import tensorflow as tf
import numpy as np
import pandas as pd


def gen_istd(img_dir,mask_dir,list_name,output_dir,out_prefix,scale=5):
    mask_list=pd.read_csv(osp.join(mask_dir,list_name))["mask"].map(
        lambda x:osp.join(mask_dir,x)).tolist()
    img_list=pd.read_csv(osp.join(img_dir,list_name))["ambient"].map(
        lambda x:osp.join(img_dir,x)).tolist()

    mask_list=tf.constant(mask_list)
    n=len(img_list)
    img_list=np.array(img_list)

    ds=tf.data.Dataset.from_tensor_slices(img_list)
    ds=ds.map(lambda x:gen_shadow2(x,mask_list),num_parallel_calls=4).prefetch(4)
    classes=["_A","_B","_C"]
    for cls in classes:
        p=osp.join(output_dir,out_prefix+cls)
        if not osp.exists(p):
            os.makedirs(p)

    for j in range(scale):
        iterator=ds.make_one_shot_iterator()
        next_imgs=iterator.get_next()
        with tf.Session() as sess:
            for i in range(n):
                imgs=sess.run(next_imgs)
                for k, cls in enumerate(classes):
                    with open(osp.join(output_dir,out_prefix+cls,"%04d-%01d.png"%(i,j)),'wb') as f:
                        f.write(imgs[k])

def write_split_list(list,sections,out_dir,out_names,columns):
    list=np.random.permutation(list)
    sub_lists=np.split(list, sections)
    for sub_list,out_name in zip(sub_lists,out_names):
        print("%s size: %d"%(out_name,len(sub_list)))
        df=pd.DataFrame(sub_list,columns=columns)
        df.to_csv(osp.join(out_dir,out_name),index=False)



def encode_jpeg(img):
    return tf.io.encode_jpeg(
                tf.image.convert_image_dtype(
                    img,tf.uint8,saturate=True))

def encode_png(img):
    return tf.image.encode_png(
                    tf.image.convert_image_dtype(
                    img,tf.uint8,saturate=True))

def rgbg2rgb(rgbg):
    return tf.stack((rgbg[...,0],(rgbg[...,1]+rgbg[...,3])/2,rgbg[...,2]),axis=2)

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

def concat_img(imgs,col=3):
    batch=tf.stack(imgs)[:,::2,::2,:]
    shape = tf.shape(batch)[0:4]
    n=shape[0]
    h=shape[1]
    w=shape[2]
    c=shape[3]
    row=n // col
    concat= tf.reshape(
            tf.transpose(
            tf.reshape( batch,
            shape=(row,col,h,w,c)),
            perm=[0,2,1,3,4]),
            shape=(row*h,col*w,c) )
    
    return concat

def load_paired_data(img_names, id):
    tmp_pureflash = imread(img_names[5 * id + 4])[None,...]/255.
    tmp_ambient = imread(img_names[5 * id])[None,...]/255.
    tmp_flash = imread(img_names[5 * id + 3])[None,...]/255.
    tmp_T = imread(img_names[5 * id + 2])[None,...]/255.
    tmp_R = imread(img_names[5 * id + 1])[None,...]/255.
    return tmp_pureflash, tmp_ambient, tmp_flash, tmp_T, tmp_R

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

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64).clip(0,1)
    img2 = img2.astype(np.float64).clip(0,1)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

def load_img(image_path,channels=0):
  return tf.image.convert_image_dtype(
      tf.image.decode_jpeg(
      tf.io.read_file(image_path),channels=channels),dtype=tf.float32)

def save_img(image_path,image):
    return tf.io.write_file(image_path,
                encode_jpeg(image))



def darken(x,xmin=0,  xmax=0.25,  ymin=0.1,  ymax=0.9,
  mu=0.05,  sig=0.025,  smax=1):
  x1 = tf.random.uniform([],xmin, xmax)
  y1 = 0.0
  x2 = 1.0
  y2 = tf.random.uniform([],ymin, tf.math.minimum(ymax,smax*(1-x1)))
  a = (y2 - y1) / (x2 - x1)  # Assume slope = const. for all channels
  x1_G = x1
  x1_R = x1_G + tf.random.normal([],mu, sig)
  x1_B = x1_G - tf.random.normal([],mu, sig)

  b = tf.reshape(tf.stack([y1 - a * x1_R, y1 - a * x1_G, y1 - a * x1_B]),(1,1,3))
  y = a * x + b
  y = tf.clip_by_value(y, 0.0, 1.0)
  return y

def load_imgs(img_paths,num=4):
    return [tf.image.resize(load_img(img_paths[i],channels=3),(1024,1024)) for i in range(num) ]
def load_four(img_paths):
    return tf.unstack(
        tf.image.resize(tf.image.random_crop(
        tf.stack([load_img(img_paths[i])for i in range(4)]),(4,1280,1280,3)),(640,640)))

def load_raw(image_path,channels=4,dtype=tf.dtypes.uint16,maximum=1024):
      return tf.cast(tf.image.decode_png(
      tf.io.read_file(image_path),channels=channels,dtype=dtype),dtype=tf.float32)/maximum
      

def load_four_raw(img_paths):
    return tf.unstack(
        tf.image.resize(tf.image.random_crop(
        tf.stack([load_raw(img_paths[i])for i in range(4)]),(4,1280,1280,4)),(640,640)))

def gen_shadow(img_paths,mask_file_list):
  mask_index=tf.random.uniform(shape=[], minval=0, maxval=tf.shape(mask_file_list)[0], dtype=tf.int32)
  mask=load_img(mask_file_list[mask_index],channels=1)
  noshad=load_img(img_paths[0],channels=3)
  flash=load_img(img_paths[1],channels=3)
  noshad,flash=tf.unstack(
      tf.image.resize(tf.image.random_crop(tf.stack(
          (noshad,flash)),(2,960,960,3)),(640,640)))
  dark=darken(noshad)
  shadow=mask*dark+(1-mask)*noshad
  return shadow,mask,noshad,flash

def gen_shadow2(img_paths,mask_file_list):
    mask_index=tf.random.uniform(shape=[], minval=0, maxval=tf.shape(mask_file_list)[0], dtype=tf.int32)
    mask=load_img(mask_file_list[mask_index],channels=1)
    noshad=load_img(img_paths,channels=3)
    noshad=tf.image.resize(tf.image.random_crop(noshad,(960,960,3)),(640,640))
    dark=darken(noshad)
    shadow=mask*dark+(1-mask)*noshad
    mask=tf.cast(tf.cast(mask+0.5,tf.uint8),tf.float32)
    return encode_png(shadow),encode_png(mask),encode_png(noshad)
