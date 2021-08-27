#! /usr/bin/python
# -*- coding: utf8 -*-
import matplotlib.pyplot as plt 
#import cv2
import os, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config

ni = 1

def modcrop(imgs, modulo):

    tmpsz = imgs.shape
    sz = tmpsz[0:2]

    h = sz[0] - sz[0]%modulo
    w = sz[1] - sz[1]%modulo
    imgs = imgs[0:h, 0:w,:]
    return imgs

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs


def DefocusDeblur():
    ## create folders to save result images
    save_dir = './Evaluations/dual_results'
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "pretrained"

    valid_ref_img_c_list = sorted(tl.files.load_file_list(path=config.TEST.folder_path_c+'/source/', regx='.*.png', printable=False))
    valid_ref_img_l_list = sorted(tl.files.load_file_list(path=config.TEST.folder_path_l+'/source/', regx='.*.png', printable=False))
    valid_ref_img_r_list = sorted(tl.files.load_file_list(path=config.TEST.folder_path_r+'/source/', regx='.*.png', printable=False))

    H = 1120
    W = 1680

    ###====================== BUILD GRAPH ===========================###  
    with tf.device('/device:GPU:0'):      
        t_image_c = tf.placeholder('float32', [1, H, W, 3], name='t_image_c')
        t_image_l = tf.placeholder('float32', [1, H, W, 3], name='t_image_l')
        t_image_r = tf.placeholder('float32', [1, H, W, 3], name='t_image_r')
        net_g = Defocus_Deblur_Net6_ds_dual(t_image_c,t_image_l,t_image_r, is_train=False, hrg = H, wrg = W)
        result = net_g.outputs

    ###########################################################
        configg = tf.ConfigProto()
        configg.gpu_options.allow_growth = True
        sess = tf.Session(config=configg)
        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/dual.npz', network=net_g)


        ###====================== PRE-LOAD DATA ===========================###        
        valid_ref_imgs_c = read_all_imgs(valid_ref_img_c_list, path=config.TEST.folder_path_c+'/source/', n_threads=10)
        valid_ref_imgs_l = read_all_imgs(valid_ref_img_l_list, path=config.TEST.folder_path_l+'/source/', n_threads=10)
        valid_ref_imgs_r = read_all_imgs(valid_ref_img_r_list, path=config.TEST.folder_path_r+'/source/', n_threads=10)
        tl.files.exists_or_mkdir(save_dir+'/')
        n_iter = 100
        if len(valid_ref_img_l_list) < 100:
            n_iter = len(valid_ref_img_l_list) 

        for imid in range(n_iter):
            valid_ref_img_c = np.expand_dims(valid_ref_imgs_c[imid],0)     
            valid_ref_img_c = tl.prepro.threading_data(valid_ref_img_c, fn=scale_imgs_fn)   # rescale to ［－1, 1]
            valid_ref_img_l = np.expand_dims(valid_ref_imgs_l[imid],0)     
            valid_ref_img_l = tl.prepro.threading_data(valid_ref_img_l, fn=scale_imgs_fn)   # rescale to ［－1, 1]
            valid_ref_img_r = np.expand_dims(valid_ref_imgs_r[imid],0)     
            valid_ref_img_r = tl.prepro.threading_data(valid_ref_img_r, fn=scale_imgs_fn)   # rescale to ［－1, 1]

            ###======================= EVALUATION =============================###
            start_time = time.time()    
            out = sess.run(result, {t_image_c: valid_ref_img_c, t_image_l: valid_ref_img_l,t_image_r: valid_ref_img_r})                
            print("took: %4.4fs" % ((time.time() - start_time)))

            print("[*] save images")
            tl.vis.save_image(out[0], save_dir+'/' + valid_ref_img_l_list[imid][0:-4] + '.png')

    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, evaluate')    
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    
    DefocusDeblur()
