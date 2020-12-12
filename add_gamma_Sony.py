# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io
import numpy as np
import rawpy
import glob

output_dir = './dataset/Sony/short_synthetic_gamma/'
gt_dir = './dataset/Sony/long/'

# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

DEBUG = 1
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]

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

in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# Unlike training phase, Raw data need not be in memory for long???
gt_images = [None] * 6000
output_image_gammas = np.random.randint(0, 3, 6000)
gammas = [100, 250, 300]
output_image_gammas = gammas[output_image_gammas] #hate this line

# Generate Gamma curve
gamma_curve = np.zeros((3, 65536), dtype=np.uint16)
g = [[]*65536]*3#65536 should be removed? put 6 instead? check g[j] below
for j in range(3):
    g[j] = [1//(gammas[j]), 4.500000, 0.081243, 0.018054, 0.099297, 0.517181]
    for i in range(65536):
        if (i/65535 < g[3]):
            gamma_curve[j][i] = int(i*g[j][1])
        else:
            gamma_curve[j][i] = int(65535*(np.power(1.0*i/65535, g[j][0])*(1+g[j][4])-g[j][4]))

cnt = 0
for ind in range(len(train_ids)):
    # get the path from image id
    train_id = train_ids[ind]
    gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
    gt_path = gt_files[0]
    gt_fn = os.path.basename(gt_path)

    out_ratio = output_image_gammas[ind]#gamma for the image
    #out_exposure = gt_exposure // out_ratio
    #in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
    #in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
    #in_fn = os.path.basename(in_path)

    st = time.time()
    cnt += 1
    gt_raw = rawpy.imread(gt_path)
    out_image = np.expand_dims(pack_raw(gt_raw), axis=0)
    print(max(out_image))
    out_ratio_gamma_curve_index = gammas.index(out_ratio)
    #out_image = gamma_curve[out_ratio_gamma_curve_index][out_image]#Is it int?

    #https://en.wikipedia.org/wiki/Rec._709#Transfer_characteristics - automatic gamma = (2.222, 4.5)?
    #https://letmaik.github.io/rawpy/api/rawpy.Params.html?highlight=gamma
    #http://goldsequence.blogspot.com/2019/11/libraw-post-processing-pipeline.html
    #im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)
    
    output = np.minimum(np.maximum(output, 0), 1)

    if epoch % save_freq == 0:
        if not os.path.isdir(result_dir + '%04d' % epoch):
            os.makedirs(result_dir + '%04d' % epoch)

        temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
        scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))
