import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import scipy.misc
import cv2
import numpy as np

def modcrop(imgs, modulo):

    tmpsz = imgs.shape
    sz = tmpsz[0:2]

    h = sz[0] - sz[0]%modulo
    w = sz[1] - sz[1]%modulo
    imgs = imgs[0:h, 0:w,:]
    return imgs

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')
#    return cv2.imread(path + file_name)

def get_imgs_fn_crop(file_name, path):
    """ Input an image path and name, return an image array """
    out = scipy.misc.imread(path + file_name, mode='RGB')    
    return modcrop(out,64)
#    return cv2.imread(path + file_name)

def get_flows_fn(file_name, path):
    flow = cv2.imread(path + file_name, cv2.IMREAD_UNCHANGED)
    flow = np.float32(flow) / 10. - (2 ** 15 - 1) / 10.
    return flow

def get_flows_fn2(file_name, path):
    flow = np.load(path + file_name)
    flow = np.concatenate((flow,np.expand_dims(flow[...,1],2)),2)
    return flow


def scale_imgs_fn_input(x):
    """for 0-1 input"""
    x = x/(255.)
#    x = x-1.

    return x
def scale_imgs_fn(x):
    x = x-127.5
    x = x/(255./2.)
    # x = x/255.
    return x

def crop_sub_imgs_fn(x, is_random=True, wrg=384, hrg=384):
    x = crop(x, wrg=wrg, hrg=hrg, is_random=is_random)
    return x

def crop_sub_frames_fn(xs, is_random=True, wrg=384, hrg=384):    
    xs = crop_multi(xs, wrg=wrg, hrg=hrg, is_random=is_random)
    return xs

def downsample_fn(x, is_random=True, fx=0.25, fy=0.25, inter=cv2.INTER_CUBIC):
    x = cv2.resize(x,None,fx=fx,fy=fy,interpolation=inter)
    return x

def upsample_fn(x, is_random=True, fx=4, fy=4, inter=cv2.INTER_CUBIC):    
    x = cv2.resize(x,None,fx=fx,fy=fy,interpolation=inter)
    return x

def blur_fn(x, ksize=(9,9), sigma=1.5):
    x = cv2.GaussianBlur(x,ksize=ksize,sigmaX=sigma)
    return x

def eablur_fn(x, radius=4, eps=0.04):
    x_copy = x.astype(np.float32) / 255.
    y = cv2.ximgproc.guidedFilter(x_copy, x_copy, radius, eps)
    return y * 255.

def noise_fn(x, is_random=True, sigma=0.05):
    row,col,ch= x.shape
    if is_random==True:
        sigma=random.random()*sigma
    mean = 0
    gauss = 255.*np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    return x + gauss