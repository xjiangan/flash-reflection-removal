import tensorflow as tf

def load_img(image_path,channels=0):
  return tf.image.decode_jpeg(
      tf.io.read_file(image_path),channels=channels)/255

def save_img(image_path,image):
    tf.io.write_file(image_path,
                tf.io.encode_jpeg(tf.cast(image,tf.uint8)))

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
  mask=load_img(mask_file_list[mask_index])
  noshad=load_img(img_paths[0])
  flash=load_img(img_paths[1])
  noshad,flash=tf.unstack(
      tf.image.resize(tf.image.random_crop(tf.stack(
          (noshad,flash)),(2,960,960,3)),(640,640)))
  dark=darken(noshad)
  shadow=mask*dark+(1-mask)*noshad
  return noshad,flash,shadow,mask

