import numpy as np

def rgbg2rgb(raw_array):
    return np.stack((raw_array[...,0],(raw_array[...,1]+raw_array[...,3])/2,raw_array[...,2]),axis=2)

def rgbg2linref(raw_array,maximum=1023-64):
    raw_array=raw_array.astype(np.float32)
    return np.clip(rgbg2rgb(raw_array)/maximum,0,1)

def lrgb2srgb(lrgb):
    return np.power(lrgb,1/2.2)

def rgbg2srgb(raw_array,gamma=2.2,maximum=65535):
    return (lrgb2srgb(linref2lrgb(rgbg2linref(raw_array,maximum=maximum)))*225).astype(np.uint8)



def linref2lrgb(linref,forward_matrix=np.array([[ 0.51215854,0.26953794,0.17924102],
                                        [ 0.10933821,0.85002241,0.04063938],
                                        [ 0.01164852 ,-0.43424423,1.24290822]]),
                           neutral_wb=np.array([0.59085778, 1. ,0.44278711])):

    d50tosrgb = np.reshape(np.array([3.1338561, -1.6168667, -0.4906146, 
                                    -0.9787684, 1.9161415, 0.0334540, 
                                     0.0719453, -0.2289914, 1.4052427]), (3,3))
    height, width, _ = linref.shape
    linref = np.minimum(linref,neutral_wb.reshape((1,1,3)))
    rgb_reshaped = linref.reshape((-1,3))
    camera2d50 = forward_matrix / neutral_wb.reshape((1,3))
    camera2srgb = d50tosrgb@camera2d50

    rgb_srgb = rgb_reshaped@ camera2srgb.T
    orgshape_rgb_srgb = np.reshape(rgb_srgb,(height, width,3))
    return orgshape_rgb_srgb.clip(0, 1)

##################    tests   ################################

# def ntr_stats(subdirs):
#     ntrs=[]
#     for root in subdirs:
#         dir1=osp.join(root,'dng')
#         ls =os.listdir(dir1)
#         for f in ls:
#             ntr=extract_exif(osp.join(dir1,f))
#             print(ntr)
#             return
#             ntrs.append(ntr.reshape((3,)))
#     ntrs=np.array(ntrs)
#     print(ntrs.mean(axis=0))
#     print(ntrs.std(axis=0))
#     print(np.histogram(ntrs[:,0]))