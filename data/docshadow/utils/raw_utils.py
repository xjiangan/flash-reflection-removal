from multiprocessing import Pool
import os
import os.path as osp

import exifread
import rawpy
import numpy as np
import imageio
import argparse
import hashlib

import cv2 as cv
import pandas as pd

def file_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def batch_md5(folder:str):
    out=osp.basename(folder)+'_md5.csv'
    filenames=os.listdir(folder)
    md5s=[file_md5(osp.join(folder,file)) for file in filenames]
    df=pd.DataFrame({"name":filenames,"md5":md5s})
    df.to_csv(out,index=False)





def diff(x,y):
    return {k: (x[k],y[k]) for k in x if k in y and str(x[k]) != str(y[k])}

def obtain_rgb_flashonly(paths,use_camera_wb=True, black_level = 64):
    """paths = ambient,flash[,gt] 
       return ambient_rgb,flash_rgb,flash_only_rgb[,gt_rgb] """ 
    exifs=[]
    for path in paths:
        with open(path,'rb') as f:
            exifs.append(exifread.process_file(f,details=True))
    for i in range(len(exifs)-1):
        assert len(diff(exifs[i],exifs[i+1]))<=3   
    raws=[rawpy.imread(path) for path in paths]
    rgbs=[raw.postprocess(use_camera_wb=use_camera_wb) for raw in raws]
    raws[1].raw_image_visible[:] = np.clip(raws[1].raw_image_visible.astype(np.int64) 
                                    - raws[0].raw_image_visible.astype(np.int64) 
                                    + black_level,0,1023).astype(np.uint16)
    rgbs.insert(2,raws[1].postprocess(use_camera_wb=use_camera_wb))
    for raw in raws:
        raw.close()

    return rgbs
def process_raws(i,paths):
    prefixs=["a","f","fo","gt"]
    root='dng'
    out_format="png"
    rgbs=obtain_rgb_flashonly([os.path.join(root,path) for path in paths])
    for j,img in enumerate(rgbs):
        imageio.imsave(os.path.join(out_format,f'{paths[0][:-4]}_{prefixs[j]}.{out_format}'), img)

def dng2format(root,path,out_format,use_camera_wb=True):
    with rawpy.imread(os.path.join(root,'dng',path)) as raw:
        img=raw.postprocess(use_camera_wb=use_camera_wb)
        imageio.imsave(os.path.join(root,out_format,f'{path[:-4]}.{out_format}'), img)



def convert(file,output,use_camera_wb=True,r=0.5):
    with rawpy.imread(file) as raw:
        img=raw.postprocess(use_camera_wb=use_camera_wb,no_auto_bright=True)
        imageio.imsave(output,cv.resize(img,None,fx=r,fy=r))


def scale(i,o,r=1/16):
    img=cv.imread(i)
    if img is None:
        print(i)
    print(img.shape)
    cv.imwrite(o,cv.resize(img,None,fx=r,fy=r))
def cvt_format(i:str,o:str):
    """convert image format based on file extension"""
    imwrite(o,imread(i))



def cvts2(root,fm1='dng',fm2='png',processes=16):
    dir1,dir2=osp.join(root,fm1),osp.join(root,fm2)
    ls1=os.listdir(dir1)
    if not os.path.exists(dir2):
        os.makedirs(dir2)
    outputs=bj(dir2,be(ls1,fm2))
    inputs=bj(dir1,ls1)

    with Pool(processes) as p:
        p.starmap_async(cvt_format, zip(inputs,outputs))
def get_flashpng(inputs,output):
    raws=[imread(path) for path in inputs[:2]]
    fo = np.clip(raws[1].astype(np.int32) 
            - raws[0].astype(np.int32),0,1023).astype(np.uint16)
    cv.imwrite(output,fo)
def getfopngc(root):
    df=pd.read_csv(osp.join(root,'trip.csv'))
    dfi=df[["ab","f"]].applymap(lambda x:osp.join(root,'pngc',x+'.png')).to_numpy().squeeze()
    out_dir=osp.join(root,'fopngc')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dfo=df[["f"]].applymap(lambda x:osp.join(out_dir,x+'.png')).to_numpy().squeeze()
    with Pool(16) as p:
        p.starmap(get_flashpng, zip(dfi,dfo))

def get_sdm_png(inputs,output):
    raws=[imread(path) for path in inputs[:2]]
    fo = np.clip(raws[1].astype(np.int32) 
            - raws[0].astype(np.int32),0,1023).astype(np.uint16)
    sdm=(np.clip((raws[1]-raws[0])/(raws[1]),0,1)*1023).astype(np.uint16)
    print(np.histogram(sdm))

    imageio.imwrite(output,sdm)
def getmpng(root):
    df=pd.read_csv(osp.join(root,'trip.csv'))
    dfi=df[["ab","gt"]].applymap(lambda x:osp.join(root,'pngc',x+'.png')).to_numpy().squeeze()
    out_dir=osp.join(root,'mpngc')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dfo=df[["ab"]].applymap(lambda x:osp.join(out_dir,x+'.png')).to_numpy().squeeze()
    with Pool(16) as p:
        p.starmap_async(get_sdm_png, zip(dfi,dfo))

def get_m(gt,ab):
    h,w,_=gt.shape
    down_scale=4
    gt,ab = [cv.resize(x,(w//down_scale,h//down_scale)).astype(np.float32) for x in (gt,ab)]
    return cv.resize(np.clip((gt-ab)/gt,0,1)*1023,(w,h)).astype(np.uint16)


def check_tri(csv):
    df=pd.read_csv(csv)


# def batch_convert(fm1,fm2):
#     ls1=os.listdir(fm1)
#     if not os.path.exists(fm2):
#         os.makedirs(fm2)
#     outputs=bj(fm2,be(ls1,fm2))
#     inputs=bj(fm1,ls1)
#     for i,o in zip(inputs,outputs):
#         print(i,o)
#         convert(i,o)

if __name__ == '__main__':
    # parser=argparse.ArgumentParser()
    # parser.add_argument('--root',type=str)
    # parser.add_argument('--cpu',type=int)
    # parser.add_argument('--format',type=str,default='jpg')

    # args=parser.parse_args()
    # if not os.path.exists(args.out):
    #     os.makedirs(args.out)
    
    # root=args.root
    # out_format=args.format
    # ls=os.listdir(os.path.join(root,'dng'))
    # with Pool(args.cpu) as p:
    #     p.starmap(process_raws, list(enumerate(ls)))
    batch_md5('dng')
