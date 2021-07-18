
from operator import sub
from imageio.core.functions import imread, imwrite
from utils.raw_utils import batch_convert,convert,scale,diff
from utils.io import imread, imwrite,get_exif,get_trip,raw2rgbg,mdwrite
from utils.isp import rgbg2srgb
from utils.path import be,bj,mkdir
from multiprocessing import Pool, RawArray
import matplotlib.pyplot as plt
import os
import os.path as osp
import pandas as pd
import numpy as np
import exifread
import imageio
import rawpy
from glob import glob
import cv2 as cv
import shutil
from skimage.metrics import peak_signal_noise_ratio
import logging
from pprint import pprint
import yaml

def tmd():
    df=pd.read_csv("name_tri.csv")
    df["fo"]=df["f"].map(lambda x: f"![x](fo32/{x}.png)")
    df[["ab","f","gt"]]=df[["ab","f","gt"]].applymap(lambda x:f"![x](32/{x}.jpg)")
    df.to_csv("four.md",sep=' ',line_terminator="  \n",index=False,header=False)

def tbnails2():
    inputs=glob("/data/xjiangan/flash-reflection-removal/data/flashAmbient/*/*ambient.png")
    out_dir="fads"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outputs=[osp.join(out_dir,osp.basename(path)) for path in inputs]

    with Pool(16) as p:
        p.starmap_async(scale, zip(inputs,outputs))
def sbumd():
    root="/data/xjiangan/sbu/SBU-shadow/SBU-Test/ShadowImages"
    ls=np.random.permutation(os.listdir(root))[:16].reshape(4,4)
    df=pd.DataFrame(data=ls).applymap(lambda x:f"![{x}]({root}/{x})")
    df.to_csv("sbu.md",sep=' ',line_terminator="  \n",index=False,header=False)
def three(root):
    df=get_trip(root).applymap(lambda x:f"![{x}](jpg16/{x+'.jpg'})")
    df.to_csv(osp.join(root,"three.md"),sep=' ',line_terminator="  \n",index=False,header=False)

def four(root,f_dir='fo16',j_dir='jpg16',out='four.md'):
    df=get_trip(root)
    df["fo"]=df["f"].map(lambda x: f"![x]({f_dir}/{x}.jpg)")
    df[["ab","f","gt"]]=df[["ab","f","gt"]].applymap(lambda x:f"![x]({j_dir}/{x}.jpg)")
    df.to_csv(osp.join(root,out),sep=' ',line_terminator="  \n",index=False,header=False)

def get_sc(root):
    df=get_trip(root)["ab"]
    if not os.path.exists(osp.join(root,'sc')):
        os.makedirs(osp.join(root,'sc'))
    for s in df:
        shutil.copy(osp.join(root,'jpgc',s+'.jpg'),
                    osp.join(root,'sc',s+'.jpg'))

def get_psnr(root,res_dir):
    df=get_trip(root)
    psnr=[]
    for _,gt,res,f in df.itertuples():
        gt=osp.join(root,'jpgc',gt+'.jpg')
        res=osp.join(root,res_dir,res+'.jpg')
        gt,res=(imageio.imread(x) for x in (gt,res))
        psnr.append(peak_signal_noise_ratio(gt,res))
    print(psnr)
    print(np.mean(psnr))
def count_ds(ls):
    n=0
    for root in ls:
        n+= len(get_trip(root))
    print(n)

def inplace_rotate(input_path):
    img=imageio.imread(input_path,format="PNG-FI")
    imageio.imwrite(input_path,np.rot90(img,3),format='PNG-FI')

def batch_rotate(root,fm1='png',num_worker=16):
    dir1=osp.join(root,fm1)
    ls1=os.listdir(dir1)
    inputs=bj(dir1,ls1)
    with Pool(num_worker) as p:
        p.map(inplace_rotate, inputs)

def ptest(a,b):
    print(a,b)

def tbnails(root,in_dir='jpg',out_dir='jpg16'):
    in_dir,out_dir=osp.join(root,in_dir),osp.join(root,out_dir)
    ls=os.listdir(in_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    inputs=bj(in_dir,ls)
    outputs=bj(out_dir,ls)

    with Pool(3) as p:
        p.starmap_async(scale, zip(inputs,outputs))
        
def get_bb(root):
    df=get_trip(root)
    gts=pd.unique(df["gt"].map(lambda x:x+'.jpg'))
    inputs=bj(osp.join(root,'jpg16'),gts)
    bbs=[]
    for i in inputs:
        img=imageio.imread(i)
        h,w,_=img.shape
        plt.imshow(img)
        plt.draw()
        # plt.imshow(img)
        pts =plt.ginput(2)
        print(pts)
        if len(pts)<2:
            break
        (x0, y0), (x1, y1)=pts
        bbs.append([y0/h,y1/h,x0/w,x1/w])
    bbs=np.array(bbs)
    print(bbs)
    bb=[np.max(bbs[:,0]),np.min(bbs[:,1]),np.max(bbs[:,2]),np.min(bbs[:,3])]
    print(bb)
    with open(osp.join(root,'bb.csv'),'a') as f:
        f.write(','.join([str(b) for b in bb]))

def gt_stat(root):
    df=get_trip(root)
    ls=df["gt"].map(lambda x:osp.join(root,'dng',x+'.dng'))
    exists=[]
    for f in ls:
        if f in exists:
            continue
        exists.append(f)
        gt=imread(f)
        print(np.histogram(gt))