import rawpy
import numpy as np
import cv2 as cv
import yaml
import exifread
import os
import os.path as osp
import pandas as pd
from utils.path import bj

def raw_read_rgb(file):
    with rawpy.imread(file) as raw:
        img=raw.postprocess(use_camera_wb=True,no_auto_bright=True)
    return cv.resize(img,None,fx=0.5,fy=0.5)

def mdwrite(path:str,df:pd.DataFrame):
    df.to_csv(path,sep=' ',line_terminator="  \n",index=False,header=False)


def parse_trip(root,img_dir='dng'):
    ls=sorted(os.listdir(osp.join(root,img_dir)))
    out=[]
    with open(osp.join(root,'num.csv'),'r')as f:
        i=0
        for line in f:
            num=int(line)
            if num==0:
                i=i+1
                continue
            gt=ls[i]
            out.extend([(gt,ls[j],ls[j+1]) for j in range(i+1,i+1+2*num,2)])
            i=i+2*num+1
    return pd.DataFrame(data=np.array(out),columns=["gt","ab","f"]).applymap(lambda x:x[:-4])

def get_trip(root):
    if os.path.exists(osp.join(root,'trip.csv')):
        df=pd.read_csv(osp.join(root,'trip.csv'))
    else:
        df=parse_trip(root)
        df.to_csv(osp.join(root,'trip.csv'),index=False)
    return df

def raw2rgbg(raw_array,cfa_pattern='rggb'):
    if cfa_pattern=='rggb':
        rgbg = np.stack([raw_array[0::2, 0::2], raw_array[0::2, 1::2] ,
                        raw_array[1::2, 1::2],raw_array[1::2, 0::2]], axis=2)
    elif cfa_pattern=='bggr':
        rgbg = np.stack([raw_array[1::2, 1::2], raw_array[0::2, 1::2] ,
                        raw_array[0::2, 0::2],raw_array[1::2, 0::2]], axis=2)

    return rgbg


def imread(path:str):
    """Read image unchanged
        Read rotated raw demosaic as RGBG - blacklevel """
    if path[-4:] == '.dng':
        exif=get_exif(path)
        black_level=exif['Image BlackLevel'].values
        white_level=exif["Image WhiteLevel"].values
        orientation_type=exif['Image Orientation'].values[0]
        cfa_pattern=exif['Image CFAPattern'].values
        white_level=1023
        black_level=64
        assert (np.array(black_level) ==black_level).all()
        assert (np.array(white_level) ==white_level).all()
        assert orientation_type ==6
        assert  cfa_pattern==[0,1,1,2]
        rotation=3
        with rawpy.imread(path) as raw:
            img=np.rot90(np.clip((
                raw2rgbg(raw.raw_image_visible,'rggb').astype(np.float32)-black_level)/(white_level-black_level)*65535,
                0,65535).astype(np.uint16),rotation)
    else:
        img=cv.imread(path,cv.IMREAD_UNCHANGED )
        if img.shape[2]==3:
            img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
        elif img.shape[2]==4:
            img=cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
        else:
            assert img.shape[2] in [3,4]
    # print(path)
    # print(img.shape)
    # print(img.max())
    return img

# def imread(path:str):
#     """Read image unchanged
#         Read rotated raw demosaic as RGBG - blacklevel """
#     if path[-4:] == '.dng':
#         exif=get_exif(path)
#         black_level=exif['Image BlackLevel'].values
#         orientation_type=exif['Image Orientation'].values[0]
#         assert (np.array(black_level) ==64).all()
#         assert orientation_type ==6
#         black_level=64
#         rotation=3
#         with rawpy.imread(path) as raw:
#             img=np.rot90(np.maximum(raw2rgbg(raw.raw_image_visible).astype(np.float32)-black_level,0).astype(np.uint16),rotation)
#     else:
#         img=cv.imread(path,cv.IMREAD_UNCHANGED )
#         if img.shape[2]==3:
#             img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
#         elif img.shape[2]==4:
#             img=cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
#         else:
#             assert img.shape[2] in [3,4]
#     # print(path)
#     # print(img.shape)
#     # print(img.max())
#     return img

def imwrite(path:str,img):
    if img.shape[2]==3:
        cv.imwrite(path,cv.cvtColor(img, cv.COLOR_RGB2BGR))
    elif img.shape[2]==4:
        cv.imwrite(path,cv.cvtColor(img, cv.COLOR_RGBA2BGRA))
    else:
        assert img.shape[2] in [3,4]

def parse_exif(exif_dict):
    with open('exiftag.yaml','r') as f:
        exiftag=yaml.load(f,Loader=yaml.FullLoader)
    for tag,key in exiftag.items():
        if tag in exif_dict:
            exif_dict["Image "+key]=exif_dict.pop(tag)
    return exif_dict

def get_exif(path):
    with open(path,'rb')as f:
        exif=parse_exif(exifread.process_file(f,details=True))
    return exif

def extract_exif(path):
    exif_dict=get_exif(path)
    image_temparatue = 4000
    # Standard light A
    temparature1 = 2850
    # D65
    temparature2 = 6500
    forward_matrix = get_matrix(tag2matrix(exif_dict['Image ForwardMatrix1']),
                                tag2matrix(exif_dict['Image ForwardMatrix2']),
                                temparature1, temparature2, image_temparatue)
    neutral_wb = tag2matrix(exif_dict['Image AsShotNeutral'])
    print(forward_matrix)
    print(neutral_wb)
    return forward_matrix

def tag2matrix(tag):
    return np.reshape([x.decimal() for x in tag.values],(3,-1))

def get_matrix(m1, m2, tp1, tp2, tp):
    if (tp < tp1):
        m = m1
    elif (tp > tp2):
        m = m2
    else:
        g = (1/ float(tp) - 1 / float(tp2)) / (1 / float(tp1) - 1 / float(tp2))
        m = g * m1 + (1-g) * m2
    return m


##################    tests   ################################
def check_shape(root,fm1='jpgc',fm2='pngc'):
    dir1,dir2=osp.join(root,fm1),osp.join(root,fm2)
    ls1,ls2=bj(dir1,sorted(os.listdir(dir1))),bj(dir2,sorted(os.listdir(dir2)))
    for f1,f2 in zip(ls1,ls2):
        im1=imread(f1)
        im2=imread(f2)
        print(im2.max(),end='\r')
        if im1.shape[:2] != im2.shape[:2]:
            print(im1.shape,im2.shape)
            return