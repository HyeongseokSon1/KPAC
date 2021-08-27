#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def Defocus_Deblur_Net6_ms(t_image, ks = 5, bs=2, ch=48, is_train=False, reuse=False, hrg=128, wrg=128, name="deblur_net"):
    w_init = tf.random_normal_initializer(stddev=0.04)
    # w_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope(name, reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n_ref = InputLayer(t_image, name='in')
        n = n_ref
        n = Conv2d(n, ch, (5, 5), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c0')  
        n = Conv2d(n, ch, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c0_2')  
        f1 = n                      
        n = Conv2d(n, ch, (3, 3), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c2')
        n = Conv2d(n, ch, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c2_2')  
        f2 = n        
        n = Conv2d(n, ch*2, (3, 3), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c3')        
        n = Conv2d(n, ch*2, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c3_2')  
        temp1 = n
        stack = n
        ## pre residual blocks
    for i in range(bs):
        with tf.variable_scope(name, reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            nn1 = AtrousConv2dLayer(n, ch, (ks, ks), rate=1, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
        with tf.variable_scope(name, reuse=True):
            tl.layers.set_name_reuse(True)
            nn2 = AtrousConv2dLayer(n, ch, (ks, ks), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
            nn3 = AtrousConv2dLayer(n, ch, (ks, ks), rate=3, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
            nn4 = AtrousConv2dLayer(n, ch, (ks, ks), rate=4, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
            nn5 = AtrousConv2dLayer(n, ch, (ks, ks), rate=5, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
            
        with tf.variable_scope(name, reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            nn = ConcatLayer([nn1,nn2,nn3,nn4,nn5], 3, 'concat/%s' % (i))
            ## scale attention (spatially varying)
            # n_sc = MeanPool2d(n, filter_size=(8, 8), strides=(8, 8), padding='SAME', name='sca_pool/%s' % (i))
            n_sc = AtrousConv2dLayer(n, 32, (5, 5), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='sca_dc1/%s' % (i))
            n_sc = AtrousConv2dLayer(n_sc, 32, (5, 5), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='sca_dc2/%s' % (i))
            n_sc = AtrousConv2dLayer(n_sc, 16, (5, 5), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='sca_dc3/%s' % (i))
            n_sc = AtrousConv2dLayer(n_sc, 16, (5, 5), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='sca_dc4/%s' % (i))
            n_sc = Conv2d(n_sc, 5, (5, 5), (1, 1), act=tf.nn.sigmoid, padding='SAME', W_init=w_init,b_init=b_init, name='sca_c2/%s' % (i))
            # n_sc.outputs = tf.nn.softmax(n_sc.outputs)
            fa = n_sc
            n_sc = TransposeLayer(n_sc, [0,1,3,2],'sca_trans/%s' % (i))
            n_sc = UpSampling2dLayer(n_sc, [1,ch], method=1, name='sca_up/%s' % (i))
            n_sc = TransposeLayer(n_sc, [0,1,3,2],'sca_trans_inv/%s' % (i))
            # n_sc = UpSampling2dLayer(n_sc, [hrg/4,wrg/4], is_scale=False, method=1, align_corners=False, name='sca_attention/%s' % (i))
            nn = ElementwiseLayer([nn, n_sc], tf.multiply, 'sca_attention_mul/%s' % (i))

            ## shape attention (global, shared)
            n_sh = MeanPool2d(n, filter_size=(hrg/4, wrg/4), strides=(hrg/4, wrg/4), padding='SAME', name='sha_pool/%s' % (i))
            n_sh = Conv2d(n_sh, ch/4, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='sha_c1/%s' % (i))
            n_sh = Conv2d(n_sh, ch, (1, 1), (1, 1), act=tf.nn.sigmoid, padding='SAME', W_init=w_init,b_init=b_init, name='sha_c2/%s' % (i))
            n_sh = TileLayer(n_sh, [1,1,1,5], name='sha_tile/%s' % (i))
            n_sh = UpSampling2dLayer(n_sh, [hrg/4,wrg/4], is_scale=False, method=1, align_corners=False, name='sha_attention/%s' % (i))
            nn = ElementwiseLayer([nn, n_sh], tf.multiply, 'sha_attention_mul/%s' % (i))

            nn = Conv2d(nn, ch*2, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='cf/%s' % (i))
            # nn = ElementwiseLayer([n, nn], tf.add, name = 'residual_add/%s' % (i))
            n = nn              
            stack = ConcatLayer([stack, n], 3, name = 'dense_concat/%s' % (i))
            # stack = n
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)     
        n = Conv2d(stack, ch*2, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/cm_1')
        n = Conv2d(n, ch*2, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/cm_2')

        n = DeConv2d(n, ch, (4, 4), (hrg/2, wrg/2), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d1')
        n = ConcatLayer([n, f2], 3, name='pre/s1')
        n = Conv2d(n, ch, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d1_2')

        n = DeConv2d(n, ch, (4, 4), (hrg, wrg), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d2')
        n = ConcatLayer([n, f1], 3, name='pre/s2')
        n = Conv2d(n, 3, (5, 5), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d2_2')
        n = ElementwiseLayer([n, n_ref], tf.add, name='post/s3')
        return n

def Defocus_Deblur_Net6_ds(t_image, ks = 5, bs = 2, is_train=False, reuse=False, hrg=128, wrg=128, name="deblur_net"):
    w_init = tf.random_normal_initializer(stddev=0.04)
    # w_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope(name, reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n_ref = InputLayer(t_image, name='in')
        n = n_ref
        n = Conv2d(n, 48, (5, 5), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c0')  
        n = Conv2d(n, 48, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c0_2')  
        f1 = n                      
        n = Conv2d(n, 48, (3, 3), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c2')
        n = Conv2d(n, 48, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c2_2')  
        f2 = n        
        n = Conv2d(n, 96, (3, 3), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c3')        
        n = Conv2d(n, 96, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c3_2')  
        f3 = n
        n = Conv2d(n, 96, (3, 3), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c4')        
        n = Conv2d(n, 96, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c4_2')  
        temp1 = n
        stack = n
        ## pre residual blocks
    for i in range(bs):
        with tf.variable_scope(name, reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            nn1 = AtrousConv2dLayer(n, 48, (ks, ks), rate=1, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
        with tf.variable_scope(name, reuse=True):
            tl.layers.set_name_reuse(True)
            nn2 = AtrousConv2dLayer(n, 48, (ks, ks), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
            nn3 = AtrousConv2dLayer(n, 48, (ks, ks), rate=3, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
            nn4 = AtrousConv2dLayer(n, 48, (ks, ks), rate=4, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
            nn5 = AtrousConv2dLayer(n, 48, (ks, ks), rate=5, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
            
        with tf.variable_scope(name, reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            nn = ConcatLayer([nn1,nn2,nn3,nn4,nn5], 3, 'concat/%s' % (i))
            ## scale attention (spatially varying)
            # n_sc = MeanPool2d(n, filter_size=(8, 8), strides=(8, 8), padding='SAME', name='sca_pool/%s' % (i))
            n_sc = AtrousConv2dLayer(n, 32, (5, 5), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='sca_dc1/%s' % (i))
            n_sc = AtrousConv2dLayer(n_sc, 32, (5, 5), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='sca_dc2/%s' % (i))
            n_sc = AtrousConv2dLayer(n_sc, 16, (5, 5), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='sca_dc3/%s' % (i))
            n_sc = AtrousConv2dLayer(n_sc, 16, (5, 5), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='sca_dc4/%s' % (i))
            n_sc = Conv2d(n_sc, 5, (5, 5), (1, 1), act=tf.nn.sigmoid, padding='SAME', W_init=w_init,b_init=b_init, name='sca_c2/%s' % (i))
            # n_sc.outputs = tf.nn.softmax(n_sc.outputs)
            fa = n_sc
            n_sc = TransposeLayer(n_sc, [0,1,3,2],'sca_trans/%s' % (i))
            n_sc = UpSampling2dLayer(n_sc, [1,48], method=1, name='sca_up/%s' % (i))
            n_sc = TransposeLayer(n_sc, [0,1,3,2],'sca_trans_inv/%s' % (i))
            # n_sc = UpSampling2dLayer(n_sc, [hrg/4,wrg/4], is_scale=False, method=1, align_corners=False, name='sca_attention/%s' % (i))
            nn = ElementwiseLayer([nn, n_sc], tf.multiply, 'sca_attention_mul/%s' % (i))

            ## shape attention (global, shared)
            n_sh = MeanPool2d(n, filter_size=(hrg/8, wrg/8), strides=(hrg/4, wrg/4), padding='SAME', name='sha_pool/%s' % (i))
            n_sh = Conv2d(n_sh, 12, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='sha_c1/%s' % (i))
            n_sh = Conv2d(n_sh, 48, (1, 1), (1, 1), act=tf.nn.sigmoid, padding='SAME', W_init=w_init,b_init=b_init, name='sha_c2/%s' % (i))
            n_sh = TileLayer(n_sh, [1,1,1,5], name='sha_tile/%s' % (i))
            n_sh = UpSampling2dLayer(n_sh, [hrg/8,wrg/8], is_scale=False, method=1, align_corners=False, name='sha_attention/%s' % (i))
            nn = ElementwiseLayer([nn, n_sh], tf.multiply, 'sha_attention_mul/%s' % (i))

            nn = Conv2d(nn, 96, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='cf/%s' % (i))
            # nn = ElementwiseLayer([n, nn], tf.add, name = 'residual_add/%s' % (i))
            n = nn              
            stack = ConcatLayer([stack, n], 3, name = 'dense_concat/%s' % (i))        
            # stack = n
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)     
        n = Conv2d(stack, 96, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/cm_1')
        n = Conv2d(n, 96, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/cm_2')

        n = DeConv2d(n, 96, (4, 4), (hrg/4, wrg/4), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d0')
        n = ConcatLayer([n, f3], 3, name='pre/s0')
        n = Conv2d(n, 96, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d0_2')

        n = DeConv2d(n, 48, (4, 4), (hrg/2, wrg/2), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d1')
        n = ConcatLayer([n, f2], 3, name='pre/s1')
        n = Conv2d(n, 48, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d1_2')

        n = DeConv2d(n, 48, (4, 4), (hrg, wrg), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d2')
        n = ConcatLayer([n, f1], 3, name='pre/s2')
        n = Conv2d(n, 3, (5, 5), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d2_2')
        n = ElementwiseLayer([n, n_ref], tf.add, name='post/s3')
        return n

def Defocus_Deblur_Net6_ds_dual(t_image_c, t_image_l, t_image_r, ks = 5, bs = 2, is_train=False, reuse=False, hrg=128, wrg=128, name="deblur_net"):
    """ Generator in Deep Video deblurring
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.04)
    # w_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope(name, reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n_ref = InputLayer(t_image_c, name='in_c')
        n_l = InputLayer(t_image_l, name='in_l')
        n_r = InputLayer(t_image_r, name='in_r')

        n = ConcatLayer([n_l,n_r], 3, name='concat_input')
        n = Conv2d(n, 48, (5, 5), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c0')  
        n = Conv2d(n, 48, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c0_2')  
        f1 = n                      
        n = Conv2d(n, 48, (3, 3), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c2')
        n = Conv2d(n, 48, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c2_2')  
        f2 = n        
        n = Conv2d(n, 96, (3, 3), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c3')        
        n = Conv2d(n, 96, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c3_2')  
        f3 = n
        n = Conv2d(n, 96, (3, 3), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c4')        
        n = Conv2d(n, 96, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='deblur/c4_2')  
        temp1 = n
        stack = n
        ## pre residual blocks
    for i in range(bs):
        with tf.variable_scope(name, reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            nn1 = AtrousConv2dLayer(n, 48, (ks, ks), rate=1, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
        with tf.variable_scope(name, reuse=True):
            tl.layers.set_name_reuse(True)
            nn2 = AtrousConv2dLayer(n, 48, (ks, ks), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
            nn3 = AtrousConv2dLayer(n, 48, (ks, ks), rate=3, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
            nn4 = AtrousConv2dLayer(n, 48, (ks, ks), rate=4, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
            nn5 = AtrousConv2dLayer(n, 48, (ks, ks), rate=5, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=None, name='dc1/%s' % (i))
            
        with tf.variable_scope(name, reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            nn = ConcatLayer([nn1,nn2,nn3,nn4,nn5], 3, 'concat/%s' % (i))
            ## scale attention (spatially varying)
            # n_sc = MeanPool2d(n, filter_size=(8, 8), strides=(8, 8), padding='SAME', name='sca_pool/%s' % (i))
            n_sc = AtrousConv2dLayer(n, 32, (5, 5), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='sca_dc1/%s' % (i))
            n_sc = AtrousConv2dLayer(n_sc, 32, (5, 5), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='sca_dc2/%s' % (i))
            n_sc = AtrousConv2dLayer(n_sc, 16, (5, 5), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='sca_dc3/%s' % (i))
            n_sc = AtrousConv2dLayer(n_sc, 16, (5, 5), rate=2, act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='sca_dc4/%s' % (i))
            n_sc = Conv2d(n_sc, 5, (5, 5), (1, 1), act=tf.nn.sigmoid, padding='SAME', W_init=w_init,b_init=b_init, name='sca_c2/%s' % (i))
            # n_sc.outputs = tf.nn.softmax(n_sc.outputs)
            fa = n_sc
            n_sc = TransposeLayer(n_sc, [0,1,3,2],'sca_trans/%s' % (i))
            n_sc = UpSampling2dLayer(n_sc, [1,48], method=1, name='sca_up/%s' % (i))
            n_sc = TransposeLayer(n_sc, [0,1,3,2],'sca_trans_inv/%s' % (i))
            # n_sc = UpSampling2dLayer(n_sc, [hrg/4,wrg/4], is_scale=False, method=1, align_corners=False, name='sca_attention/%s' % (i))
            nn = ElementwiseLayer([nn, n_sc], tf.multiply, 'sca_attention_mul/%s' % (i))

            ## shape attention (global, shared)
            n_sh = MeanPool2d(n, filter_size=(hrg/8, wrg/8), strides=(hrg/4, wrg/4), padding='SAME', name='sha_pool/%s' % (i))
            n_sh = Conv2d(n_sh, 12, (1, 1), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='sha_c1/%s' % (i))
            n_sh = Conv2d(n_sh, 48, (1, 1), (1, 1), act=tf.nn.sigmoid, padding='SAME', W_init=w_init,b_init=b_init, name='sha_c2/%s' % (i))
            n_sh = TileLayer(n_sh, [1,1,1,5], name='sha_tile/%s' % (i))
            n_sh = UpSampling2dLayer(n_sh, [hrg/8,wrg/8], is_scale=False, method=1, align_corners=False, name='sha_attention/%s' % (i))
            nn = ElementwiseLayer([nn, n_sh], tf.multiply, 'sha_attention_mul/%s' % (i))

            nn = Conv2d(nn, 96, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init,b_init=b_init, name='cf/%s' % (i))
            # nn = ElementwiseLayer([n, nn], tf.add, name = 'residual_add/%s' % (i))
            n = nn              
            stack = ConcatLayer([stack, n], 3, name = 'dense_concat/%s' % (i))        
            # stack = n
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)     
        n = Conv2d(stack, 96, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/cm_1')
        n = Conv2d(n, 96, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/cm_2')

        n = DeConv2d(n, 96, (4, 4), (hrg/4, wrg/4), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d0')
        n = ConcatLayer([n, f3], 3, name='pre/s0')
        n = Conv2d(n, 96, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d0_2')

        n = DeConv2d(n, 48, (4, 4), (hrg/2, wrg/2), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d1')
        n = ConcatLayer([n, f2], 3, name='pre/s1')
        n = Conv2d(n, 48, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d1_2')

        n = DeConv2d(n, 48, (4, 4), (hrg, wrg), (2, 2), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d2')
        n = ConcatLayer([n, f1], 3, name='pre/s2')
        n = Conv2d(n, 3, (5, 5), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='deblur/d2_2')
        n = ElementwiseLayer([n, n_ref], tf.add, name='post/s3')
        return n



def Vgg19_simple_api2(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        start_time = time.time()
        print("build model started")
        rgb = tf.maximum(0.0,tf.minimum(rgb,1.0))        
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else: # TF 1.0
            # print(rgb_scaled)
            
            red, green, blue = tf.split(rgb_scaled, 3, 3)
#        assert red.get_shape().as_list()[1:] == [224, 224, 1]
#        assert green.get_shape().as_list()[1:] == [224, 224, 1]
#        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
#        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv2_2')
        conv2 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool3')
        conv3 = network
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv4_4')
        conv4 = network
        
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool4')                               # (batch_size, 14, 14, 512)
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                    strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv5_4')
        conv5 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                    padding='SAME', name='pool5')                               # (batch_size, 7, 7, 512)
        """ fc 6~8 """
#        network = FlattenLayer(network, name='flatten')
#        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
#        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
#        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv4