# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir_gamma = './gamma_checkpoint/Sony/'
result_dir = './result_Sony_with_gamma_net/'

# get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')#1 means test
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

DEBUG = 1
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]


def lrelu(x):
    return tf.maximum(x * 0.2, x)

def network(input):
    #https://github.com/google-research/tf-slim
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')
    #16X16 SIZE IMAGE?
    conv10 = slim.conv2d(conv5, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    flatten1 = slim.flatten(conv10)
    #out = tf.depth_to_space(conv10, 2)#channels: 12 to 3 conversion
    #out = tf.depth_to_space(out, 3)#3 to 1 channels #Made a mistake here with block_size = 1 instead of 3
    #
    #ValueError: Dimension size must be evenly divisible by 9 but is 3 for 
    #'DepthToSpace_1' (op: 'DepthToSpace') with input shapes: [?,?,?,3].
    #print(out.shape)
    #(?, ?, ?, 3)
    #out = tf.reshape(out, [1, -1])#1D#We don't have to reshape at all!!!!!!!!
    #print(out.shape)
    #(?,)
    fc1 = slim.fully_connected(flatten1, 1000, scope='fc_1')
    fc2 = slim.fully_connected(fc1, 3, scope='fc_2')
    #ValueError: Input 0 of layer fc_1 is incompatible with the layer: : expected 
    #min_ndim=2, found ndim=1. Full shape received: [None]
    #ValueError: The last dimension of the inputs to `Dense` should be defined. Found `None`.
    #https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
    return fc2

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])#4 channels coming from pack_raw()??
out_gamma = network(in_image)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir_gamma)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    #ValueError: The passed save_path is not a valid checkpoint: ./result_Sony__[1-5]_images/model.ckpt

pred_gamma_path = result_dir + 'predicted_gamma.txt'
pred_gamma_file = open(pred_gamma_path, 'a')

for test_id in test_ids:
    # test the first image in each sequence
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
    for k in range(len(in_files)):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        print(in_fn)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0)
        input_full = np.minimum(input_full, 1.0)
        
        output = sess.run(out_gamma, feed_dict={in_image: input_full})
        output = np.minimum(np.maximum(output, 0.001), 1000)
        

        #print(output)
        #print(tf.shape(output))#tf.shape(x) is dynamic shape of x, x.get_shape() gives static shape (.as_list())
        #print(output.get_shape())
        output = output[0, 0, 0, :]#This is how you get the number out!!!!! .item() didn't work for me.
        pred_gamma_file.write("%05d\t%.3f\t%.3f\n" % (test_id, ratio, output.item()))
        
pred_gamma_file.close()
