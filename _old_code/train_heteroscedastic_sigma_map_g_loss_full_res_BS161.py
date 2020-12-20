

# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division, print_function
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
from math import log
from show import *
import re
#from resnet import *


print("\n\n\n")
import datetime
now = datetime.datetime.now()
print ("Current date and time : ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))

#input_dir = './dataset/Sony/short/'
gt_long_dir = './dataset/Sony/long/'
gt_short_dir = './dataset/Sony/short/'
checkpoint_dir = './heteroscedastic_sigma_map_g_loss_full_res_BS161checkpoint/'
result_dir = checkpoint_dir
RAM_ALLOCATED = 60000#in MB
RAM_PER_IMAGE = 75.9#80640/1060

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

def get_exposure_time(file_name):
    e = re.search('0\.[0-9]+',file_name)
    #print(e)
    #print(type(e))
    return e.group(0)

#The file name contains the image information. For example, in "10019_00_0.033s.RAF",
#the first digit "1" means it is from the test set ("0" for training set and 
#"2" for validation set); "0019" is the image ID; the following "00" is 
#the number in the sequence/burst; "0.033s" is the exposure time 1/30 seconds.

# get train IDs

train_fns = glob.glob(gt_long_dir + '0*[0-9]_*_*.ARW')#0 in the beginning means training set
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]
#train_exposures = [float(get_exposure_time(os.path.basename(train_fn))) for train_fn in train_fns]
for i in range(7):
    print(train_fns[i], train_ids[i])#, train_exposures[i])

print("Found " + str(len(train_ids)) + " images to train with\n")

ps = 128  # patch size for training
save_freq = 500

DEBUG = 1
if DEBUG == 1:
    save_freq = 1
    init_index = 0
    total_images = len(train_fns)
    limit = int(RAM_ALLOCATED / RAM_PER_IMAGE) - 10
    if(limit < total_images):
        init_index = (np.random.randint(0, total_images - limit, 1)).item()
    limit = min(limit, total_images)
    #init_index = 0
    #limit = 16
    print("Training on files with indices: " + str(init_index) + " to " + str(init_index+limit))
    train_ids = train_ids[init_index:(init_index+limit)]
    train_fns = train_fns[init_index:(init_index+limit)]
    #train_exposures = train_exposures[init_index:(init_index+limit)]

print("Training on " + str(len(train_ids)) + " images only\n")


def lrelu(x):
    return tf.maximum(x * 0.2, x)

def relu(x):
    return tf.maximum(x * 0.0, x)

def network(input):
    #https://github.com/google-research/tf-slim
    
    #HAVE TO ADD Regularizer LOSS
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=relu, scope='g_conv1_1')
    conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=relu, scope='g_conv2_1')

    conv3 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=relu, scope='g_conv3_1')
    conv4 = slim.conv2d(conv3, 32, [3, 3], rate=1, activation_fn=relu, scope='g_conv4_1')
    
    conv10 = slim.conv2d(conv4, 2, [3, 3], rate=1, activation_fn=relu, scope='g_conv10')
    return conv10

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
noise_level_map = tf.placeholder(tf.int32, [None, None, None, 2])# batch size = 1 in this paper, BTW
out_noise_level_map = network(in_image)
#gt_gamma_one_hot = tf.squeeze(slim.one_hot_encoding(gt_gamma, 3), axis=1)
#gt_gamma_one_hot = tf.Print(gt_gamma_one_hot, [gt_gamma_one_hot])
#G_loss = tf.losses.softmax_cross_entropy(gt_gamma_one_hot, out_gamma)
G_loss = tf.losses.mean_squared_error(noise_level_map, out_noise_level_map)
#G_loss = tf.losses.huber_loss(gt_noise_sigmas, out_noise_sigmas)
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
else:
    print('No checkpoint found at ' + checkpoint_dir + '. Hence, will create the folder.')

#4240x2832 for Sony and 6000x4000 for the Fuji images.
input_images_numpy = np.zeros((len(train_ids), 1424, 2128, 4))
for_sony = 1
if(for_sony ==  1):
    H= 1424
    W = 2128
g_loss = np.ones((len(train_fns), H, W, 2))
print(np.shape(g_loss))

allfolders = glob.glob(result_dir + '*_results')#why is there a zero at the end? save_freq could be 1 before.
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-12:-8]))
print("\nlast epoch of previous run: " + str(lastepoch))

st = time.time()
cnt = 0

BATCH_SIZE = len(train_fns)
no_of_batches = len(train_fns)//BATCH_SIZE
input_patch = np.zeros((BATCH_SIZE, ps, ps, 4))

init_time = time.time()
learning_rate = 1e-4
final_epoch = 4001#lastepoch + 1 #4001

print("\n\nBATCH_SIZE=", BATCH_SIZE, ",final_epoch=", final_epoch, ",no_of_batches=", no_of_batches, 
      ",ps", ps, ",len(train_fns)=", len(train_fns), ",\nresult_dir=", result_dir)

input_images = [None] * len(train_fns)
input_images_status = [0] * len(train_fns)
images_in_memory = [0]
def clear_input_images(input_images, input_images_status, images_in_memory):
    for i in range(len(train_fns)):
        input_images[i] = 0
        input_images_status[i] = 0
    images_in_memory[0] = 0
    print("\nCleared all images in memory.\n")
    
def clear_input_images_partial(input_images, input_images_status, images_in_memory, count):
    deleted_count = 0
    for i in range(len(train_fns)):
        if(deleted_count < count):
            if(input_images_status[i] == 2):
                input_images[i] = 0
                input_images_status[i] = 0
                deleted_count += 1
                images_in_memory[0] -= 1 
    print("\nCleared " + str(deleted_count) + " images in memory. (This can be done smarter.)")    

clear_input_images(input_images, input_images_status, images_in_memory)


input_image= None#used in getting H after initializing in the training loop. Be careful if moving.

st = time.time()
#Train with images
for epoch in range(lastepoch, final_epoch):
    if os.path.isdir(result_dir + '%04d_results' % epoch):
        continue
    if epoch > 1000:
        learning_rate = 1e-5

    cnt = 0
    epoch_init_time= time.time()
    random_iter = np.random.randint(0, len(train_fns), no_of_batches * BATCH_SIZE)
    random_iter = random_iter.reshape((no_of_batches, BATCH_SIZE))
    for batch_id in range(no_of_batches):     
        ind = random_iter[batch_id]
        #ind = np.array(train_ids)[ind]
        if(DEBUG == 1 and epoch == lastepoch and cnt == 0):
            print("Starting Training on index " + str(ind))
            print("dataset index: " + str(np.array(train_fns)[ind]))
        #exposures = np.array(train_exposures)[ind]#exposure for the image
        #if(DEBUG == 1 and epoch == lastepoch and cnt == 0):
            #print("Starting Training on exposures " + str(exposures))
        
        if(images_in_memory[0] > int(RAM_ALLOCATED / RAM_PER_IMAGE)- 10 ):
            clear_input_images_partial(input_images, input_images_status, images_in_memory, 100)#Cqn be made more EFFICIENT
        #Load the images so that GPU doesn't wait
        for index in ind:
            #Load if it's not in memory
            #desired_image = input_images[index]
            if(input_images_status[index] == 0):#THIS CONDITION IS AMBIGUOUS?
                if(DEBUG == 1 and index % 10 == 0 and epoch < lastepoch + 10):
                    print("loading " + train_fns[index] + "; images_in_memory= " + str(images_in_memory[0]))
                #if(DEBUG==1 and epoch - lastepoch > 10):
                #    if(g_loss[index] == 0):
                #        print("loading image " + train_fns[index] + "for the first time in this run.")
                # get the path from image id
                train_id = train_ids[index]
                train_fn = train_fns[index]
                #exposure = train_exposures[index]
                gt_fn = os.path.basename(train_fn)
                gt_raw = rawpy.imread(train_fn)
                if(DEBUG == 1 and index % 10 == 1 and epoch == lastepoch):
                    print("rawpy read the " + str(index) + "th file at location: " + str(gt_fn))
                input_image = pack_raw(gt_raw)
                
                input_images[index] = input_image
                input_images_status[index] = 2
                images_in_memory[0] += 1
            #else:
            #    if(DEBUG==1  and index % 200 == 1 and epoch < lastepoch + 10):
            #        print("Found in memory: " + train_fns[index] + "; images_in_memory= " + str(images_in_memory[0]))
        st = time.time()
        cnt += 1

        # crop
        H = input_image.shape[0]
        W = input_image.shape[1]
        D = input_image.shape[2]
        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        
        for k in range(BATCH_SIZE):
            #THIS INVERSION CAUSES ANY PROBLEMS?
            input_patch[k,:,:,:] = input_images[ind[k]][yy:yy + ps, xx:xx + ps, :]


        if(DEBUG == 1 and (epoch - lastepoch) % 200 == 0 and cnt == 1):
            print("\n\n(mean,stddev), image[0], hetero noise[0], const noise[0], image[0]")
            for k in range(BATCH_SIZE):
                print(np.mean(input_patch[k]), np.std(input_patch[k]))
            print(input_patch[0,ps//2:ps//2+2,ps//2:ps//2+2,1])
        #print(np.shape(input_patch))
        #print(np.shape(g_loss))
        #Add noise - following Le Ciu's paper or (Yanghai Tsin's paper)
        #Could try different noises for each imageXXXXXXXXXXXXXXXXXX
        heteroschedastic_sigma_s_max = 0.16#Chosen from Le Ciu's
        sigma_c_max = 0.06#Chosen from Le Ciu's
        heteroschedastic_sigma_s = np.random.uniform(0, heteroschedastic_sigma_s_max, 1)
        heteroschedastic_sigma_s_map = np.full((BATCH_SIZE,ps,ps,1), heteroschedastic_sigma_s)
        heteroschedastic_noise_s = np.random.normal(0, 1, (BATCH_SIZE,ps,ps,D)) * heteroschedastic_sigma_s
        input_patch = np.multiply(input_patch, np.add(1.0,  heteroschedastic_noise_s))
        sigma_c = np.random.uniform(0,sigma_c_max)
        sigma_c_map = np.full((BATCH_SIZE,ps,ps,1), sigma_c)
        noise_c = np.random.normal(0, 1, (BATCH_SIZE,ps,ps,D)) * sigma_c
        input_patch = np.add(input_patch, noise_c)
        if(DEBUG == 1 and (epoch - lastepoch) % 200 == 0 and cnt == 1):
            print("heteroschedastic_sigma_s", heteroschedastic_sigma_s)
            print("sigma_c", sigma_c)
            print(np.multiply(heteroschedastic_noise_s[0,ps//2:ps//2+2,ps//2:ps//2+2,1], 
                             input_patch[0,ps//2:ps//2+2,ps//2:ps//2+2,1]))
            print(noise_c[0,ps//2:ps//2+2,ps//2:ps//2+2,1], "\n")
        

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
        #if np.random.randint(10, size=1)[0] > 2:#0.7 probability
        #    input_patch[:, np.random.randint(0,ps,1),:,:] = 1#Jittering to 0 will not change much
        #if np.random.randint(10, size=1)[0] > 2:
        #    input_patch[:, :, np.random.randint(0,ps,1),:] = 1#Jittering to 0 will not change much
        
        
        input_patch = np.clip(input_patch, 0.0, 1.0)
        noise_sigma_map = np.concatenate((heteroschedastic_sigma_s_map, sigma_c_map), axis = 3)
        noise_sigmas_feed = noise_sigma_map
        
        
        #print(np.shape(exposures_feed))
        _, G_current, output = sess.run([G_opt, G_loss, out_noise_level_map],
                                        feed_dict={in_image: input_patch, noise_level_map: noise_sigmas_feed, lr: learning_rate})
        #output = np.minimum(np.maximum(output, 0.0001), 1000)#bounds for gamma
        #g_loss[ind] = G_current#This is wrong if the batch size > 1
        for k in range(BATCH_SIZE):
            g_loss[ind[k], yy:yy + ps, xx:xx + ps, :] = abs((noise_sigmas_feed[k] - output[k]))
            #g_loss[ind[k], 0,0, :] = abs(np.mean(np.mean((noise_sigmas_feed[k] - output[k]), axis = 1), axis=1))
            #g_loss[ind[k],yy:yy + ps, xx:xx + ps,:] = abs(noise_sigmas_feed[k] - output[k])
            #print(k)
            #if(DEBUG == 1 and (epoch - lastepoch) % 100 == 1 and cnt == 1):
            #    if(k == 0):
            #        print("Delta of, sigmas, estimated sigmas:")
            #    print(g_loss[ind[k]])
            #    print(noise_sigmas_feed[k])
            #    print(output[k], "\n")

        #moving_loss = moving_loss_alpha*moving_loss + (1 - moving_loss_alpha)*np.mean(g_loss[np.where(g_loss)]).item()
        if(cnt == 1 and ((epoch - lastepoch) % 10 == 1 or (epoch - lastepoch) < 10) ):
            training_dataset_loss = np.mean(g_loss[np.where(g_loss)])
            print("Epoch %d: at batch %d: Training dataset Loss=%.6f, Batch Time=%.3f" % (
                epoch, cnt, training_dataset_loss, time.time() - st))
        if(cnt < 1000 and ((epoch - lastepoch) < 3) ):
            training_dataset_loss = np.mean(g_loss[np.where(g_loss)])
            print("Epoch %d: at batch %d: Training dataset Loss=%.6f, Batch Time=%.3f; Early rounds" % (
                epoch, cnt, training_dataset_loss, time.time() - st))

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d_results' % epoch):
                os.mkdir(result_dir + '%04d_results' % epoch)
            file_path = result_dir + ('%04d_results/' % epoch) + 'intermediate_results.txt'
            text_file = open(file_path, 'a')
                #print(tf.shape(assigned_image_gamma_index))#Tensor("Shape:0", shape=(0,), dtype=int32)
            text_file.write('\nEpoch %04d\tBatch %05d\n' % (epoch, batch_id))
            #output_list = output * scale
            #np.savetxt(text_file, output_list)
            #exposures_list = exposures.tolist()
            #exposures_list = [str(x[0]) for x in exposures_list]
                #assigned_image_gamma_list = map(str, assigned_image_gamma_list)
            #s = ", "
            #s = s.join(exposures_list)
            #text_file.write(s)
                #trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                #with tf.variable_scope('conv1', reuse=True):
                    #weights_conv_1 = tf.get_variable('weights')
            #text_file.write("\nScaled loss (mean of batch):\n")
            #g_loss_list = g_loss[ind]
            #g_loss_list = [str(x[0]) for x in g_loss_list]
                #assigned_image_gamma_list = map(str, assigned_image_gamma_list)
            #s = ", "
            #s = s.join(g_loss_list)
            #text_file.write(s)
            text_file.close()
            
            
    if((epoch - lastepoch) % 200 == 1 or (epoch - lastepoch) < 3):
        print("\t\tEpoch %d:  Time = %.3f, Avg epoch time=%.3f, Current epoch Time=%.3f\n" % (
            epoch, time.time() - init_time, time.time() - epoch_init_time, 
            (time.time() - init_time) / (epoch - lastepoch + 1)))
    if((epoch - lastepoch) % 200 == 2 or (epoch - lastepoch) < 3):
        print("Loss vector (slice for the first 10 images)")
        print(g_loss[0:min(10,len(train_fns)), yy:yy + 2, xx:xx + 2,:])#just 2x2 pixels
        
    #if((epoch - lastepoch) % 100 == 99):
        #validate_model()
    saver.save(sess, checkpoint_dir + 'model.ckpt')
    
    