

# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division, print_function
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
from pip._vendor.pkg_resources import null_ns_handler
from math import log
from show import *
import tf.contrib.gan.eval.image_grid as image_grid

print("\n\n\n")
import datetime
now = datetime.datetime.now()
print ("Current date and time : ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))

#input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir = './gt_Sony_CNN5_FC2_no_log_jitter/'
result_dir = checkpoint_dir

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

#The file name contains the image information. For example, in "10019_00_0.033s.RAF",
#the first digit "1" means it is from the test set ("0" for training set and 
#"2" for validation set); "0019" is the image ID; the following "00" is 
#the number in the sequence/burst; "0.033s" is the exposure time 1/30 seconds.

# get train IDs
train_fns = glob.glob(gt_dir + '0*[0-9]_00_*.ARW')#0 in the beginning means training set
#/opt/slurm/data/slurmd/job14037828/slurm_script: line 21:  7915 Killed                  python train_for_gamma_Sony.py
#slurmstepd: error: Detected 1 oom-kill event(s) in step 14037828.batch cgroup. Some of your processes may have been killed by the cgroup out-of-memory handler.
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

print("Found " + str(len(train_ids)) + " images to train with\n")

ps = 128  # patch size for training
save_freq = 500

DEBUG = 1
if DEBUG == 1:
    save_freq = 10
    #train_ids = train_ids[0:16]

print("Training on " + str(len(train_ids)) + " images only\n")

def lrelu(x):
    return tf.maximum(x * 0.2, x)

def relu(x):
    return tf.maximum(x * 0.0, x)

def network(input):
    #https://github.com/google-research/tf-slim
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=relu, scope='g_conv1_1')
    bn1 = slim.batch_norm(conv1, scope='g_conv1_bn1')
    conv1 = slim.conv2d(bn1, 32, [3, 3], rate=1, activation_fn=relu, scope='g_conv1_2')
    bn1 = slim.batch_norm(conv1, scope='g_conv1_bn2')
    pool1 = slim.max_pool2d(bn1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=relu, scope='g_conv2_1')
    bn2 = slim.batch_norm(conv2, scope='g_conv2_bn1')
    conv2 = slim.conv2d(bn2, 64, [3, 3], rate=1, activation_fn=relu, scope='g_conv2_2')
    bn2 = slim.batch_norm(conv2, scope='g_conv2_bn2')
    pool2 = slim.max_pool2d(bn2, [2, 2], padding='SAME')

    #conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    #bn2 = slim.batch_norm(conv2, scope='g_conv1_bn2')
    #conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')
    #bn2 = slim.batch_norm(conv2, scope='g_conv1_bn2')
    
    conv10 = slim.conv2d(pool2, 12, [1, 1], rate=1, activation_fn=relu, scope='g_conv10')
    bn10 = slim.batch_norm(conv10, scope='g_conv10_bn1')
    flatten1 = slim.flatten(bn10)
    flatten1.set_shape([None, 12*32*32])
    fc1 = slim.fully_connected(flatten1, 1000, scope='fc_1')
    bn1_fc = slim.batch_norm(fc1, scope='g_fc1_bn1')
    fc2 = slim.fully_connected(bn1_fc, 1000, scope='fc_2')
    bn2_fc = slim.batch_norm(fc2, scope='g_fc2_bn1')
    fc3 = slim.fully_connected(bn2_fc, 1, scope='fc_3')
    return fc3

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


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])#4 channels coming from pack_raw()??
gt_gamma = tf.placeholder(tf.int32, [None, 1])#Just one gamma number per image. batch size = 1 in this paper, BTW
out_gamma = network(in_image)
#gt_gamma_one_hot = tf.squeeze(slim.one_hot_encoding(gt_gamma, 3), axis=1)
#gt_gamma_one_hot = tf.Print(gt_gamma_one_hot, [gt_gamma_one_hot])
#G_loss = tf.losses.softmax_cross_entropy(gt_gamma_one_hot, out_gamma)
G_loss = tf.losses.mean_squared_error(gt_gamma, out_gamma)
#G_loss = tf.losses.absolute_difference(gt_gamma_one_hot, out_gamma)

t_vars = tf.trainable_variables()
lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('\nLoaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    #ValueError: The passed save_path is not a valid checkpoint: ./result_Sony__[1-5]_images/model.ckpt
else:
    print('No checkpoint found at ' + checkpoint_dir + '. Hence, will create the folder.')
    


# Unlike training phase, Raw data need not be in memory for long???
gt_images = [None] * len(train_ids)#6000 is a number bigger than the dataset size
assigned_image_gamma_indices = np.random.randint(0, 3, len(train_ids))
gammas = [100, 250, 300]
assigned_image_gammas = np.array(gammas)[assigned_image_gamma_indices] #hate this line
#TypeError: only integer scalar arrays can be converted to a scalar index 
#-> This forced gammas to be converted to nparray

input_images = [None] * len(train_ids)
#4240x2832 for Sony and 6000x4000 for the Fuji images.
input_images_numpy = np.zeros((len(train_ids), 1424, 2128, 4))


g_loss = np.zeros((len(train_ids), 1))

allfolders = glob.glob(result_dir + '*0')#why is there a zero at the end? save_freq could be 1 before.
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))
print("\nlast epoch of previous run: " + str(lastepoch))

st = time.time()
cnt = 0

#Load all images so that GPU doesn't wait
for ind in range(len(train_ids)):
    # get the path from image id
    train_id = train_ids[ind]

    gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
    gt_path = gt_files[0]
    gt_fn = os.path.basename(gt_path)
    
    gt_raw = rawpy.imread(gt_path)
    if(DEBUG == 1 and ind % 10 == 0):
        print("rawpy read the " + str(ind) + "th file at location: " + str(gt_path))
    input_image = pack_raw(gt_raw)
    #input_image = np.expand_dims(pack_raw(gt_raw), axis=0)#adding dimension because they use only one image in batch

    assigned_image_gamma = assigned_image_gammas[ind]
    input_image = np.multiply(input_image, 1./assigned_image_gamma)
    
    input_images_numpy[ind] = input_image
    cnt += 1
    #if(ind ==0):
    if(ind < 20):
        print("min, max, mean, gamma, argmax: %.5f, %.5f, %.5f, %.5f, %d" % (np.min(input_image), np.max(input_image), np.mean(input_image), assigned_image_gamma, np.argmax(input_image)))
print("%d images loaded to CPU RAM in Time=%.3f seconds." % (cnt, time.time() - st))

print("\nMoved images data to a numpy array.")
#train_dataset = tf.data.Dataset.from_tensor_slices((input_images, assigned_image_gammas))
#train_dataset = train_dataset.repeat(epoch - lastepoch)
BATCH_SIZE = 16
no_of_batches = len(train_ids)//BATCH_SIZE
input_patch = np.zeros((BATCH_SIZE, ps, ps, 4))
#train_dataset = train_dataset.batch(BATCH_SIZE)
#iterator = train_dataset.make_one_shot_iterator()
#moving_loss = 0
#moving_loss_alpha = 0.99

init_time = time.time()
learning_rate = 1e-4
final_epoch = 4001#lastepoch + 1 #4001

print("\n\n\nBATCH_SIZE", BATCH_SIZE, ",final_epoch", final_epoch, ",no_of_batches", no_of_batches, 
      ",ps", ps, ",result_dir", result_dir, ",len(train_ids)", len(train_ids))
print("Scaling the log regression labels now.\n")

st = time.time()
#Train with images
for epoch in range(lastepoch, final_epoch):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    cnt = 0
    if epoch > 1000:
        learning_rate = 1e-5

    epoch_init_time= time.time()
    random_iter = np.random.randint(0, len(train_ids), no_of_batches * BATCH_SIZE)
    random_iter = random_iter.reshape((no_of_batches, BATCH_SIZE))
    for batch_id in range(no_of_batches):
        ind = random_iter[batch_id]
        #ind = np.array(train_ids)[ind]
        if(DEBUG == 1 and epoch == lastepoch and cnt == 0):
            print("Starting Training on index " + str(ind) + "\ndataset index: " + str(np.array(train_ids)[ind]))
        assigned_image_gamma = assigned_image_gammas[ind]#gamma for the image
        if(DEBUG == 1 and epoch == lastepoch and cnt == 0):
            print("Starting Training on gammas " + str(assigned_image_gamma))
        #assigned_image_gamma_index = assigned_image_gamma_indices[ind]#Should be same as in assigned_image_gamma_indices
        #out_image = gamma_curve[assigned_image_gamma_index][out_image]#Is it int?
        
        st = time.time()
        cnt += 1

        # crop
        H = input_image.shape[0]
        W = input_image.shape[1]
        D = input_image.shape[2]
        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        
        for k in range(BATCH_SIZE):
            input_patch[k,:,:,:] = input_images_numpy[ind[k], yy:yy + ps, xx:xx + ps, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
        if np.random.randint(10, size=1)[0] > 2:#0.7 probability
            input_patch[:, np.random.randint(0,ps,1),:,:] = 1#Jittering to 0 will not change much
        if np.random.randint(10, size=1)[0] > 2:
            input_patch[:, :, np.random.randint(0,ps,1),:] = 1#Jittering to 0 will not change much

        input_patch = np.minimum(input_patch, 1.0)

        assigned_image_gamma = np.transpose(np.array([assigned_image_gamma]))#conversion for the sake of gt_gamma
        assigned_image_gamma_feed = assigned_image_gamma / 300#scaling for the sake of Neural Network training
        _, G_current, output = sess.run([G_opt, G_loss, out_gamma],
                                        feed_dict={in_image: input_patch, gt_gamma: assigned_image_gamma_feed, lr: learning_rate})
        #output = np.minimum(np.maximum(output, 0.0001), 1000)#bounds for gamma
        g_loss[ind] = G_current

        #moving_loss = moving_loss_alpha*moving_loss + (1 - moving_loss_alpha)*np.mean(g_loss[np.where(g_loss)]).item()
        if(cnt == 1 and ((epoch - lastepoch) % 10 == 1 or (epoch - lastepoch) < 10) ):
            training_dataset_loss = np.mean(g_loss[np.where(g_loss)])
            print("Epoch %d: at batch %d: Training dataset Loss=%.3f, Batch Time=%.3f" % (
                epoch, cnt, training_dataset_loss, time.time() - st))

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.mkdir(result_dir + '%04d' % epoch)
            file_path = result_dir + ('%04d/' % epoch) + 'intermediate_results.txt'
            text_file = open(file_path, 'a')
            #print(tf.shape(assigned_image_gamma_index))#Tensor("Shape:0", shape=(0,), dtype=int32)
            text_file.write('\nEpoch %04d\tBatch %05d\n' % (epoch, batch_id))
            output_list = output * 300
            np.savetxt(text_file, output_list)
            assigned_image_gamma_list = assigned_image_gamma.tolist()
            assigned_image_gamma_list = [str(x[0]) for x in assigned_image_gamma_list]
            #assigned_image_gamma_list = map(str, assigned_image_gamma_list)
            s = ", "
            s = s.join(assigned_image_gamma_list)
            text_file.write(s)
            text_file.close()
            #trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            with tf.variable_scope('conv1', reuse=True):
                weights_conv_1 = tf.get_variable('weights')
                grid = image_grid(weights_conv_1, grid_shape = (4, 8), image_shape = (ps, ps), num_channels = 4)
                
            
    if((epoch - lastepoch) % 300 == 1):
        print("\t\tEpoch %d: Epoch time = %.3f, Avg epoch time=%.3f, Total Time=%.3f\n" % (
            epoch, time.time() - init_time, time.time() - epoch_init_time, 
            (time.time() - init_time) / (epoch - lastepoch + 1)))
    if((epoch - lastepoch) % 100 == 1 or (epoch - lastepoch) < 5):
        print("Loss vector (slice for the first 20 images)")
        print(g_loss[0:min(20,len(train_ids))])
        
    #if((epoch - lastepoch) % 100 == 99):
        #validate_model()
    saver.save(sess, checkpoint_dir + 'model.ckpt')
    
    