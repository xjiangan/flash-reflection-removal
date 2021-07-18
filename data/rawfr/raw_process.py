import os
import os.path as osp
import yaml
from operator import concat, sub
from utils.raw_utils import convert,scale,diff
from utils.io import imread, imwrite,get_exif,get_trip,raw2rgbg,mdwrite,raw_read_rgb,check_exif,extract_exif
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
from glob import glob
from utils.io import parse_tripr
subdirs=['huawei_20200918', 'huawei_20200910', 'huawei_20200826', 'huawei_20200909', 'huawei_20200917','nikon_20200616', 'nikon_20200617']

def get_trips():
    for input_folder in subdirs:
        df=parse_tripr(input_folder)
        df.to_csv(osp.join(input_folder,'trip.csv'),index=False)
bsd=None
def get5rgb(inputs,outputs,black_level=256):
    """inputs: ref, ab, f
        outputs: ref, ab, f, fo, tran"""
    check_exif(inputs)
    rgbs=[raw_read_rgb(x) for x in inputs]
    h,w,c=rgbs[0].shape

    for out,rgb in zip(outputs[:3],rgbs):
        imwrite(out,rgb)
    # return
    
    raws=[rawpy.imread(x) for x in inputs[1:3]]
    raws[1].raw_image_visible[:] = np.maximum(raws[1].raw_image_visible.astype(np.int64) 
                                        - raws[0].raw_image_visible.astype(np.int64) 
                                        + black_level,0).astype(np.uint16)
    fo = cv.resize(raws[1].postprocess(use_camera_wb=True,no_auto_bright=True),None,fx=0.5,fy=0.5)
    imwrite(outputs[3],fo)

    raws=[rawpy.imread(x) for x in inputs[:2]]
    raws[1].raw_image_visible[:] = np.maximum(raws[1].raw_image_visible.astype(np.int64) 
                                        - raws[0].raw_image_visible.astype(np.int64) 
                                        + black_level,0).astype(np.uint16)
    tran = cv.resize(raws[1].postprocess(use_camera_wb=True,no_auto_bright=True),None,fx=0.5,fy=0.5)
    imwrite(outputs[4],tran)

def batch5rgb(root:str,processes=16):
    df=get_trip(root)
    dfi=df.applymap(lambda x:osp.join(root,'raw',x+'.dng')).to_numpy()
    out_dirs=[osp.join(root,'rgb','origin'),osp.join(root,'rgb','derived')]
    for d in out_dirs:
        mkdir(d)
    dfo=df.applymap(lambda x:osp.join(out_dirs[0],x+'.jpg'))
    dfo[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:osp.join(out_dirs[1],x+'.jpg'))
    dfo=dfo.to_numpy()

    # get5rgb(dfi[0],dfo[0])
    # return

    with Pool(16) as p:
        ares = p.starmap_async(get5rgb, zip(dfi,dfo))
        ares.wait()

def concat_bbs(root):
    dfbd=pd.read_csv(osp.join(root,root+'.csv'))
    dftrip=pd.read_csv(osp.join(root,'trip.csv'))
    dfbb=pd.concat([dftrip,dfbd],axis=1)
    dfbb.to_csv(osp.join(root,'bb.csv'),index=False)

from pprint import pformat
def exif_dbg():
    path="nikon_20200616/raw/DSC_7550.NEF"
    a=get_exif(path)
    with open("exifnk.yaml",'w')as f:
        f.write(pformat(a))

def tbnail_rgb(inputs,outputs):
    r=1/16
    for i,o in zip(inputs,outputs):
        im=cv.imread(i)
        if im is None:
            print(i)
            continue
        # print(im.shape,im.dtype)
        cv.imwrite(o,cv.resize(im,None,fx=r,fy=r))

def batch_rgbtb(root,fmt='rgbc'):
    df=get_trip(root)
    in_dirs=[osp.join(root,fmt,'origin'),osp.join(root,fmt,'derived')]
    dfi=df.applymap(lambda x:osp.join(in_dirs[0],x+'.jpg'))
    dfi[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:osp.join(in_dirs[1],x+'.jpg'))
    dfi=dfi.to_numpy()

    out_dirs=[osp.join(root,fmt,'thumbnail','origin'),osp.join(root,fmt,'thumbnail','derived')]
    for d in out_dirs:
        mkdir(d)
    dfo=df.applymap(lambda x:osp.join(out_dirs[0],x+'.jpg'))
    dfo[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:osp.join(out_dirs[1],x+'.jpg'))
    dfo=dfo.to_numpy()

    with Pool(16) as p:
        ares = p.starmap_async(tbnail_rgb, zip(dfi,dfo))
        ares.wait()
def five(root,imformat='rgb'):
    df=get_trip(root)
    out_dirs=[osp.join(imformat,'thumbnail','origin'),osp.join(imformat,'thumbnail','derived')]
    dfo=df.applymap(lambda x:f"![{x}]({osp.join(out_dirs[0],x+'.jpg')})")
    dfo[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:f"![{x}]({osp.join(out_dirs[1],x+'.jpg')})")
    mdwrite(osp.join(root,f'five{imformat}.md'),dfo)

def crop_rgb(inputs,outputs,bds):
    w_start, h_start,w_end, h_end =[int(x) for x in bds[:4]]
    # print(h_start, w_start, h_end, w_end)
    h_offset = (h_end-h_start)//32 * 32
    w_offset = (w_end-w_start)//32 * 32
    for i,o in zip(inputs,outputs):
        img=cv.imread(i)
        # img=cv.resize(img,None,fx=0.5,fy=0.5)
        cv.imwrite(o,img[h_start:h_start+h_offset,w_start:w_start+w_offset])

def batch_crop_rgb(root):
    df=get_trip(root)
    in_dirs=[osp.join(root,'rgb','origin'),osp.join(root,'rgb','derived')]
    dfi=df.applymap(lambda x:osp.join(in_dirs[0],x+'.jpg'))
    dfi[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:osp.join(in_dirs[1],x+'.jpg'))
    dfi=dfi.to_numpy()

    out_dirs=[osp.join(root,'rgbc','origin'),osp.join(root,'rgbc','derived')]
    for d in out_dirs:
        mkdir(d)
    dfo=df.applymap(lambda x:osp.join(out_dirs[0],x+'.jpg'))
    dfo[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:osp.join(out_dirs[1],x+'.jpg'))
    dfo=dfo.to_numpy()

    dfbd=(pd.read_csv(osp.join(root,root+'.csv'))[["w1","h1","w2","h2"]]).to_numpy()
    # print(dfi[50])
    # crop_rgb(dfi[50],dfo[50],dfbd[50])
    # return
    with Pool(16) as p:
        ares = p.starmap_async(crop_rgb, zip(dfi,dfo,dfbd))
        ares.wait()

def tbnail_raw(inputs,outputs):
    r=1/16
    imgs=[cv.resize(rgbg2srgb(imread(x),maximum=65535),None,fx=r,fy=r) for x in inputs[:5]]
    for img,o in zip(imgs,outputs[:5]):
        imwrite(o,img)

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

    with Pool(16) as p:
        ares = p.starmap_async(tbnail_raw, zip(dfi,dfo))
        ares.wait()

def get5rawc(inputs,outputs,bds):
    """inputs: gt, ab, f
        outputs: gt, ab, f, fo, m"""
    raws=[imread(x) for x in inputs]
    w_start, h_start,w_end, h_end =[int(x) for x in bds[:4]]
    # print(h_start, w_start, h_end, w_end)
    h_offset = (h_end-h_start)//32 * 32
    w_offset = (w_end-w_start)//32 * 32
    rawsc=[x[h_start:h_start+h_offset,w_start:w_start+w_offset] for x in raws]

    for out,raw in zip(outputs[:3],rawsc):
        imwrite(out,raw)
    fo = np.maximum(rawsc[2].astype(np.int32) 
            - rawsc[1].astype(np.int32),0).astype(np.uint16)
    imwrite(outputs[3],fo)

    tran = np.maximum(rawsc[1].astype(np.int32) 
            - rawsc[0].astype(np.int32),0).astype(np.uint16)
    imwrite(outputs[4],tran)
    

def batch5rawc(root:str,processes=16):
    df=get_trip(root)
    dfi=df.applymap(lambda x:osp.join(root,'raw',x+'.dng')).to_numpy()
    out_dirs=[osp.join(root,'rawc','origin'),osp.join(root,'rawc','derived')]
    for d in out_dirs:
        mkdir(d)
    dfo=df.applymap(lambda x:osp.join(out_dirs[0],x+'.png'))
    dfo[["fo","m"]]=df[["f","ab"]].applymap(
            lambda x:osp.join(out_dirs[1],x+'.png'))
    dfo=dfo.to_numpy()
    dfbd=(pd.read_csv(osp.join(root,root+'.csv'))[["w1","h1","w2","h2"]]).to_numpy()

    # get5rawc(dfi[0],dfo[0],dfbd[0])
    # return
    with Pool(16) as p:
        ares = p.starmap_async(get5rawc, zip(dfi,dfo,dfbd))
        ares.wait()
    



def main3():
    for subdir in [ 'huawei_20200918', 'huawei_20200910', 'huawei_20200826']:
        batch5rawc(subdir)
        batch_rawtb(subdir)
        five(subdir,'rawc')
def main():
    sizes=[]
    for root in [ 'huawei_20200918', 'huawei_20200910', 'huawei_20200826']:
        root=osp.join(root,'rgbc','origin')
        ls=os.listdir(root)
        for f in ls:
            sizes.append(cv.imread(osp.join(root,f)).shape)
    sizes=np.array(sizes)
    # np.savez('size.npz',sizes)
    print(np.histogram(sizes[:,0]))
    print(np.histogram(sizes[:,1]))


def main2():
    for subdir in ['huawei_20200918', 'huawei_20200910', 'huawei_20200826', 'huawei_20200909', 'huawei_20200917']:
    # for subdir in ['huawei_20200917']:
        # batch5rgb(subdir)
        # batch_crop_rgb(subdir)
        # batch_rgbtb(subdir)
        # five(subdir,'rgbc')
        # concat_bbs(subdir)
        print(subdir)
        df=get_trip(subdir)
        print(len(df))
if __name__ == '__main__':
    main2()
    # dfi=["huawei_20200917/raw/IMG_20200917_153703.dng",
    # "huawei_20200917/raw/IMG_20200917_153710.dng",
    # "huawei_20200917/raw/IMG_20200917_153724.dng"]
    # dfo=["huawei_20200917/rgb/origin/IMG_20200917_153703.jpg",
    # "huawei_20200917/rgb/origin/IMG_20200917_153710.jpg",
    # "huawei_20200917/rgb/origin/IMG_20200917_153724.jpg",
    # "huawei_20200917/rgb/derived/IMG_20200917_153724.jpg",
    # "huawei_20200917/rgb/derived/IMG_20200917_153710.jpg"]
    # get5rgb(dfi,dfo)
    # a=rawpy.imread("huawei_20200917/raw/IMG_20200917_153703.dng").raw_image_visible.shape
    # a=imread("huawei_20200917/raw/IMG_20200917_153703.dng").shape
    # print(a)
