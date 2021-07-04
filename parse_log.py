import pandas as pd
import os.path as osp
import os
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('model')
args=parser.parse_args()
print(vars(args))
root=osp.join('result',args.model)
ls=[x for x in sorted(os.listdir(root)) if len(x)==4 and x!='0000']
psnr=[]
for d in ls:
    log=osp.join(root,d,'psnr_ssim.txt')
    if not os.path.exists(log):
        continue
    df=pd.read_csv(log,sep=':',names=["id","psnr"])
    try:
        psnr.append(df.tail(1)["psnr"].item())
    except ValueError:
        print("value error")
        continue
print(np.max(psnr))
print(np.argmax(psnr))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(psnr)
fig.savefig(osp.join(root,'psnr.png'))