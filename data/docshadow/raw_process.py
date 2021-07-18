
from operator import sub
from utils.raw_utils import convert,scale,diff
from utils.io import imread, imwrite,get_exif,get_trip,raw2rgbg,mdwrite,raw_read_rgb
from utils.isp import rgbg2linref, rgbg2srgb,rgbg2rgb
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
import cv2 as cv
from glob import glob
import shutil
from skimage.metrics import peak_signal_noise_ratio
import logging
from pprint import pprint
import yaml

def cvts(root,fm1='dng',fm2='jpg',num_worker=3):
    dir1,dir2=osp.join(root,fm1),osp.join(root,fm2)
    ls1=os.listdir(dir1)
    mkdir(dir2)
    outputs=bj(dir2,be(ls1,fm2))
    inputs=bj(dir1,ls1)
    with Pool(num_worker) as p:
        p.starmap_async(convert, zip(inputs,outputs))


def check_exif(inputs):
    exifs=[]
    for path in inputs:
        with open(path,'rb') as f:
            # TODO assert black level =64
            exifs.append(exifread.process_file(f,details=True))
    for i in range(len(exifs)-1):
        assert len(diff(exifs[i],exifs[i+1]))<=3  



def get_flashs(use_camera_wb=True, black_level = 64):
    df=pd.read_csv("name_tri.csv")
    dfi=df[["ab","f"]].applymap(lambda x:osp.join('dng',x+'.dng')).to_numpy().squeeze()
    out_dir='fo'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dfo=df[["f"]].applymap(lambda x:osp.join(out_dir,x+'.png')).to_numpy().squeeze()
    with Pool(16) as p:
        p.starmap_async(get_flash, zip(dfi,dfo))

def getfo(root):
    df=get_trip(root)
    dfi=df[["ab","f"]].applymap(lambda x:osp.join(root,'dng',x+'.dng')).to_numpy().squeeze()
    out_dir=osp.join(root,'fo')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dfo=df[["f"]].applymap(lambda x:osp.join(out_dir,x+'.jpg')).to_numpy().squeeze()
    with Pool(3) as p:
        p.starmap_async(get_flash, zip(dfi,dfo))

def get_sdm(inputs,output):
    check_exif(inputs)
    raws=[raw2rgbg(rawpy.imread(path).raw_image_visible) for path in inputs[:2]]
    sdm=np.rot90(np.clip((raws[1]-raws[0])/(raws[1]),0,1)*255,3,(0,1)).astype(np.uint8)

    imageio.imwrite(output,sdm)
def getm(root):
    df=get_trip(root)
    dfi=df[["ab","gt"]].applymap(lambda x:osp.join(root,'dng',x+'.dng')).to_numpy().squeeze()
    out_dir=osp.join(root,'m')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dfo=df[["ab"]].applymap(lambda x:osp.join(out_dir,x+'.jpg')).to_numpy().squeeze()
    with Pool(3) as p:
        p.starmap_async(get_sdm, zip(dfi,dfo))

def five(root,f_dir='fo16',j_dir='jpg16',m_dir='m16',out='five.md'):
    df=get_trip(root)
    df["fo"]=df["f"].map(lambda x: f"![x]({f_dir}/{x}.jpg)")
    df["m"]=df["ab"].map(lambda x:f"![x]({m_dir}/{x}.jpg)")
    df[["ab","f","gt"]]=df[["ab","f","gt"]].applymap(lambda x:f"![x]({j_dir}/{x}.jpg)")
    df.to_csv(osp.join(root,out),sep=' ',line_terminator="  \n",index=False,header=False)


############################
def bcrop(root,in_dir='jpg16',out_dir="jpg16c"):
    in_dir,out_dir=osp.join(root,in_dir),osp.join(root,out_dir)
    ls=os.listdir(in_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    inputs=bj(in_dir,ls)
    outputs=bj(out_dir,ls)

    with Pool(3) as p:
        p.starmap_async(crop, zip(inputs,outputs))



def main1():
    subdir='bio2'
    # cvts(subdir)
    # tbnails(subdir)
    ###############################################
    # trip(subdir)
    # three(subdir)

    # # #############################
    # getfo(subdir)
    # tbnails(subdir,'fo','fo16')
    # getm(subdir)
    # tbnails(subdir,'m','m16')
    # five(subdir,'fo16','jpg16','m16','five.md')
    ##############################
    # get_bb(subdir)
    #############################
    # bcrop(subdir,'jpg16','jpg16c')
    # bcrop(subdir,'m16','m16c')
    # bcrop(subdir,'fo16','fo16c')
    # five(subdir,'fo16c','jpg16c','m16c','fivec.md')

    # #######################
    # bcrop(subdir,'jpg','jpgc')
    # bcrop(subdir,'fo','foc')
    # bcrop(subdir,'m','mc')
    ##############################
    # get_sc(subdir)
    # get_psnr(subdir,'vis')


def crop(input_path,output_path,other=1):
    img=imread(input_path)
    h,w,_=img.shape
    bd=[int(h*bds[0]),int(h*bds[1]),int(w*bds[2]),int(w*bds[3])]
    if img is None:
        print(bd)
    cv.imwrite(output_path,img[bd[0]:bd[1],bd[2]:bd[3]])

def bcrop2(root,in_dir='jpg16',out_dir="jpg16c"):
    in_dir,out_dir=osp.join(root,in_dir),osp.join(root,out_dir)
    ls=os.listdir(in_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    inputs=bj(in_dir,ls)
    outputs=bj(out_dir,ls)

    with Pool(16,initialize_bds,(root,)) as p:
        p.starmap_async(crop, zip(inputs,outputs))


bds=None
def initialize_bds(root):
    global bds
    with open(osp.join(root,'bb.csv'))as f:
        bds=[float(x) for x in f.readline().split(',')]

tbnail_scale=16
def initialize_tbnail(r):
    global tbnail_scale
    tbnail_scale = r

def tbnail_raw(inputs,outputs):
    r=1/tbnail_scale
    imgs=[cv.resize(rgbg2srgb(imread(x),maximum=65535),None,fx=r,fy=r) for x in inputs[:4]]
    for img,o in zip(imgs,outputs[:4]):
        imwrite(o,img)
    imwrite(outputs[4],cv.resize(rgbg2linref(imread(inputs[4]),maximum=65535)*255,None,fx=r,fy=r).astype(np.uint8) )

def batch_rawtb(root):
    df=get_trip(root)
    in_dirs=[osp.join(root,'rawc','origin'),osp.join(root,'rawc','derived')]
    dfi=df.applymap(lambda x:osp.join(in_dirs[0],x+'.png'))
    dfi[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:osp.join(in_dirs[1],x+'.png'))
    dfi=dfi.to_numpy()

    out_dirs=[osp.join(root,'rawc','thumbnail','origin'),osp.join(root,'rawc','thumbnail','derived')]
    for d in out_dirs:
        mkdir(d)
    dfo=df.applymap(lambda x:osp.join(out_dirs[0],x+'.jpg'))
    dfo[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:osp.join(out_dirs[1],x+'.jpg'))
    dfo=dfo.to_numpy()

    with Pool(16,initialize_tbnail,(16,)) as p:
        ares = p.starmap_async(tbnail_raw, zip(dfi,dfo))
        ares.wait()
def tbnail_rgb(inputs,outputs):
    for i,o in zip(inputs,outputs):
        im=imread(i)
        if im is None:
            print(i)
            continue
        # print(im.shape,im.dtype)
        imwrite(o,cv.resize(im,None,fx=1/16,fy=1/16))

def batch_rgbtb(root):
    df=get_trip(root)
    in_dirs=[osp.join(root,'rgbc','origin'),osp.join(root,'rgbc','derived')]
    dfi=df.applymap(lambda x:osp.join(in_dirs[0],x+'.jpg'))
    dfi[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:osp.join(in_dirs[1],x+'.jpg'))
    dfi=dfi.to_numpy()

    out_dirs=[osp.join(root,'rgbc','thumbnail','origin'),osp.join(root,'rgbc','thumbnail','derived')]
    for d in out_dirs:
        mkdir(d)
    dfo=df.applymap(lambda x:osp.join(out_dirs[0],x+'.jpg'))
    dfo[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:osp.join(out_dirs[1],x+'.jpg'))
    dfo=dfo.to_numpy()

    with Pool(16,initialize_tbnail,(16,)) as p:
        ares = p.starmap_async(tbnail_rgb, zip(dfi,dfo))
        ares.wait()

def five(root,imformat='raw'):
    df=get_trip(root)
    out_dirs=[osp.join(imformat+'c','thumbnail','origin'),osp.join(imformat+'c','thumbnail','derived')]
    dfo=df.applymap(lambda x:f"![{x}]({osp.join(out_dirs[0],x+'.jpg')})")
    dfo[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:f"![{x}]({osp.join(out_dirs[1],x+'.jpg')})")
    mdwrite(osp.join(root,f'five{imformat}.md'),dfo)

def get_flash(inputs,output,use_camera_wb=True, black_level = 64):
    check_exif(inputs)
    raws=[rawpy.imread(path) for path in inputs[:2]]
    raws[1].raw_image_visible[:] = np.clip(raws[1].raw_image_visible.astype(np.int64) 
                                        - raws[0].raw_image_visible.astype(np.int64) 
                                        + black_level,0,1023).astype(np.uint16)
    img=raws[1].postprocess(use_camera_wb=use_camera_wb,no_auto_bright=True)
    print(img.shape)
    r=0.5
    imageio.imwrite(output,cv.resize(img,None,fx=r,fy=r))



def get5rawc(inputs,outputs):
    """inputs: gt, ab, f
        outputs: gt, ab, f, fo, m"""
    raws=[imread(x) for x in inputs]
    h,w,_=raws[0].shape
    bd=[int(h*bds[0]),int(h*bds[1]),int(w*bds[2]),int(w*bds[3])]
    rawsc=[x[bd[0]:bd[1],bd[2]:bd[3]] for x in raws]

    for out,raw in zip(outputs[:3],rawsc):
        imwrite(out,raw)
    fo = np.clip(rawsc[2].astype(np.int32) 
            - rawsc[1].astype(np.int32),0,65535).astype(np.uint16)
    imwrite(outputs[3],fo)
    m=(np.clip((rawsc[0].astype(np.float32)-rawsc[1].astype(np.float32))/(rawsc[0]+1),0,1)*65535).astype(np.uint16)
    imwrite(outputs[4],m)
    

def batch5rawc(root:str,processes=16):
    df=get_trip(root)
    dfi=df.applymap(lambda x:osp.join(root,'dng',x+'.dng')).to_numpy()
    out_dirs=[osp.join(root,'rawc','origin'),osp.join(root,'rawc','derived')]
    for d in out_dirs:
        mkdir(d)
    dfo=df.applymap(lambda x:osp.join(out_dirs[0],x+'.png'))
    dfo[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:osp.join(out_dirs[1],x+'.png'))
    dfo=dfo.to_numpy()
    
    # initialize_bds(root)
    # get5rawc(dfi[0],dfo[0])
    # return

    with Pool(16,initialize_bds,(root,)) as p:
        ares = p.starmap_async(get5rawc, zip(dfi,dfo))
        ares.wait()
    
def get5rgbc(inputs,outputs,black_level=64):
    """inputs: gt, ab, f
        outputs: gt, ab, f, fo, m"""
    check_exif(inputs)
    rgbs=[raw_read_rgb(x) for x in inputs]
    h,w,c=rgbs[0].shape
    bd=[int(h*bds[0]),int(h*bds[1]),int(w*bds[2]),int(w*bds[3])]
    rgbsc=[x[bd[0]:bd[1],bd[2]:bd[3]] for x in rgbs]


    for out,rgb in zip(outputs[:3],rgbsc):
        imwrite(out,rgb)
    # return
    
    raws=[rawpy.imread(x) for x in inputs[1:3]]
    raws[1].raw_image_visible[:] = np.clip(raws[1].raw_image_visible.astype(np.int64) 
                                        - raws[0].raw_image_visible.astype(np.int64) 
                                        + black_level,0,1023).astype(np.uint16)
    fo = cv.resize(raws[1].postprocess(use_camera_wb=True,no_auto_bright=True),None,fx=0.5,fy=0.5)
    imwrite(outputs[3],fo[bd[0]:bd[1],bd[2]:bd[3]])

    rgbs=[rgbg2rgb(imread(x))  for x in inputs[:2]]
    m=(np.clip((rgbs[0].astype(np.float32)-rgbs[1].astype(np.float32))/(rgbs[0]+1),0,1)*255).astype(np.uint8)
    imwrite(outputs[4],m[bd[0]:bd[1],bd[2]:bd[3]])

def batch5rgbc(root:str,processes=16):
    df=get_trip(root)
    dfi=df.applymap(lambda x:osp.join(root,'dng',x+'.dng')).to_numpy()
    out_dirs=[osp.join(root,'rgbc','origin'),osp.join(root,'rgbc','derived')]
    for d in out_dirs:
        mkdir(d)
    dfo=df.applymap(lambda x:osp.join(out_dirs[0],x+'.jpg'))
    dfo[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:osp.join(out_dirs[1],x+'.jpg'))
    dfo=dfo.to_numpy()

    # initialize_bds(root)
    # get5rgbc(dfi[0],dfo[0])
    # return

    with Pool(16,initialize_bds,(root,)) as p:
        ares = p.starmap_async(get5rgbc, zip(dfi,dfo))
        ares.wait()

def main2():
    for subdir in ['bio3','bio4','bio5','bio6','exam2','exam3','exam4']:
        get_trip(subdir)
    #     batch5rawc(subdir)
    #     batch_rawtb(subdir)
    #     five_raw(subdir)
        # subdir='bio3'
    # for subdir in ['bio','bio2','exam','scratch','calculus']:
    #     batch5rawc(subdir)
    #     batch_rawtb(subdir)
    #     five(subdir,'raw')

def main3():
    subdir='bio3'
    batch_rawtb(subdir)


def test_isp(root):
    dir_in=osp.join(root,'dng')
    
if __name__=='__main__':
    main2()
    # exifs=[get_exif(x) for x in ["calculus/dng/IMG_20210619_133408.dng","calculus/dng/IMG_20210619_134436.dng"]]
    # pprint(diff(*exifs))
    # print(tag2matrix(exifs[0]["Image AsShotNeutral"]))
    # extract_exif("calculus/dng/IMG_20210619_133408.dng")
    # test_write()

    

    