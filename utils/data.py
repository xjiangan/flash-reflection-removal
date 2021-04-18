import tensorflow as tf
import numpy as np

def encode_img(img):
    return tf.io.encode_jpeg(
                tf.image.convert_image_dtype(
                    img,tf.uint8,saturate=True))

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
                encode_img(image))

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


def gen_shadow(img_paths,mask_file_list):
  mask_index=tf.random.uniform(shape=[], minval=0, maxval=tf.shape(mask_file_list)[0], dtype=tf.int32)
  mask=load_img(mask_file_list[mask_index],channels=3)
  noshad=load_img(img_paths[0],channels=3)
  flash=load_img(img_paths[1],channels=3)
  noshad,flash=tf.unstack(
      tf.image.resize(tf.image.random_crop(tf.stack(
          (noshad,flash)),(2,960,960,3)),(640,640)))
  dark=darken(noshad)
  shadow=mask*dark+(1-mask)*noshad
  return shadow,mask,noshad,flash

