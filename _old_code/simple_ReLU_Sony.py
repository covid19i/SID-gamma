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

print("\n\n\n")


#input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir = './gt_Sony_simple_ReLU_BN_batched/'
result_dir = './gt_Sony_simple_ReLU_BN_batched/'

#The file name contains the image information. For example, in "10019_00_0.033s.RAF",
#the first digit "1" means it is from the test set ("0" for training set and 
#"2" for validation set); "0019" is the image ID; the following "00" is 
#the number in the sequence/burst; "0.033s" is the exposure time 1/30 seconds.

# get train IDs
train_fns = glob.glob(gt_dir + '0*[0-9]_00_*.ARW')#0 means training set
#/opt/slurm/data/slurmd/job14037828/slurm_script: line 21:  7915 Killed                  python train_for_gamma_Sony.py
#slurmstepd: error: Detected 1 oom-kill event(s) in step 14037828.batch cgroup. Some of your processes may have been killed by the cgroup out-of-memory handler.
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

print("Found " + str(len(train_ids)) + " images to train with\n")

ps = 128  # patch size for training
save_freq = 500

DEBUG = 1
if DEBUG == 1:
    save_freq = 10
    train_ids = train_ids[0:20]

print("Training on " + str(len(train_ids)) + " images only\n")

def lrelu(x):
    return tf.maximum(x * 0.0, x)


def network(input):
    #https://github.com/google-research/tf-slim
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    bn1 = slim.batch_norm(conv1, scope='g_conv1_bn1')
    pool1 = slim.max_pool2d(bn1, [2, 2], padding='SAME')
    conv2 = slim.conv2d(pool1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    bn2 = slim.batch_norm(conv2, scope='g_conv1_bn2')
    pool2 = slim.max_pool2d(bn2, [2, 2], padding='SAME')
    #16X16 SIZE IMAGE?
    conv10 = slim.conv2d(pool2, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv2_1')
    bn3 = slim.batch_norm(conv10, scope='g_conv2_bn1')
    flatten1 = slim.flatten(bn3)
    #print(tf.shape(flatten1))
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
    flatten1.set_shape([None, 12*32*32])
    fc1 = slim.fully_connected(flatten1, 3, scope='fc_1')
    #fc2 = slim.fully_connected(fc1, 3, scope='fc_2')
    #ValueError: Input 0 of layer fc_1 is incompatible with the layer: : expected 
    #min_ndim=2, found ndim=1. Full shape received: [None]
    #ValueError: The last dimension of the inputs to `Dense` should be defined. Found `None`.
    #https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
    return fc1


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

#https://stackoverflow.com/questions/53033767/how-to-print-a-tensors-value-inside-tf-while-loop-without-returning-it
#out_gamma = tf.Print(out_gamma, [out_gamma])
#TypeError: Expected list for 'data' argument to 'Print' Op, 
#not Tensor("fc_2/Relu:0", shape=(?, 3), dtype=float32).
#tf.Print("out_gamma: ", out_gamma)#, output_stream=sys.stdout)
#https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/print
#tf.print("out_gamma: ", out_gamma)#, output_stream=sys.stdout)
#G_loss = slim.losses.softmax_cross_entropy(out_gamma, tf.squeeze(slim.one_hot_encoding(gt_gamma, 3)))
#To remvoe the warning
#gt_gamma_copy = tf.placeholder(tf.int32, [None, 1])
#gt_gamma_copy = tf.Print(gt_gamma, [gt_gamma])
#Dangling nodes don't get computed
gt_gamma_one_hot = tf.squeeze(slim.one_hot_encoding(gt_gamma, 3), axis=1)
#gt_gamma_one_hot = tf.Print(gt_gamma_one_hot, [gt_gamma_one_hot])
#G_loss = tf.losses.softmax_cross_entropy(tf.squeeze(slim.one_hot_encoding(gt_gamma, 3)), out_gamma)
G_loss = tf.losses.softmax_cross_entropy(gt_gamma_one_hot, out_gamma)

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
    print('No checkpoint found at ' + checkpoint_dir + '. Hence, will create')
    


# Unlike training phase, Raw data need not be in memory for long???
gt_images = [None] * len(train_ids)#6000 is a number bigger than the dataset size
assigned_image_gamma_indices = np.random.randint(0, 3, len(train_ids))
gammas = [100, 250, 300]
assigned_image_gammas = np.array(gammas)[assigned_image_gamma_indices] #hate this line
#TypeError: only integer scalar arrays can be converted to a scalar index 
#-> This forced gammas to be converted to nparray

input_images = [None] * len(train_ids)


def generate_gamma_curve(gammas):
    # Generate Gamma curve
    #https://github.com/legendzhangn/blog/blob/master/libraw_isp_python/isp_simple.py
    gamma_curve = np.zeros((3, 65536), dtype=np.uint16)
    g = [[]*6]*3#65536 should be removed? put 6 instead? check g[j] below
    for j in range(3):
        #g[j] = [1/(gammas[j]), 4.500000, 0.081243, 0.018054, 0.099297, 0.517181]
        #g[j] = [1/gammas[j], 1.099297 * (np.power(0.018054, 1.0 /gammas[j]) - 0.099297)/0.018054, 
         #       0.081243, 0.018054, 0.099297, 0.517181]
        encoding_gamma = 1 / gammas[j]
        a = np.power(0.018054, 1.0 / encoding_gamma)
        print("\na: %.10f" % (a))
        b = 0.018054 / encoding_gamma
        print("b: %.10f" % (b))
        nu = -(a - b)/(a - b - 1)
        print(nu)
        eta = 1 + nu
        alpha = eta / encoding_gamma
        print(alpha)
        g[j] = [1/encoding_gamma, alpha, alpha * 0.018054, 0.018054, nu, 0.517181]
        for i in range(65536):
            gamma_curve[j][i] = int(65535 * (np.power(i/65535., g[j][0])))
            #if (i/65535 < g[j][3]):#forgot j here
                #gamma_curve[j][i] = int(i/g[j][1])
                #To add noise,
                #gamma_curve[j][i] = int(i * g[j][1])
            #else:
                #gamma_curve[j][i] = int(65535 * (np.power(i/65535., g[j][0])*(1+g[j][4]) - g[j][4]))
                #To add noise
                #gamma_curve[j][i] = int(65535 * (np.power((1.0 * i / 65535. + g[j][4]) / (1.0+g[j][4]), 
                 #                                         gammas[j]) ))
                #gamma_curve[j][i] = int(65535 * (np.power((1.0 * i / 65535. + g[j][4]) / (1.0+g[j][4]), 
                #                                          1/ g[j][0]) ))
                #np.power is taking a number instead of a numpy array as input
    print("Gamma curve:")
    #print(gamma_curve[:, 0:3])
    #print(gamma_curve[:, 990:993])
    #print(gamma_curve[:, 10000:10003])
    #print(gamma_curve[:, 34000:34003])
    #print(gamma_curve[:, 64000:64003])
    print("Every 10, 0 to 100")
    print(gamma_curve[:, 0:100:10])
    print("Every 100, 0 to 1000")
    print(gamma_curve[:, 0:1000:100])
    print("Every 1000, 0 to 8000")
    print(gamma_curve[:, 0:8000:1000])
    print("Every 8K, 0 to 64K")
    print(gamma_curve[:, 0:64000:8000])
    print("Every 1000, 57536 to 65536")
    print(gamma_curve[:, 65536-8000:65536:1000])
    print("Every 125, 64536 to 65536")
    print(gamma_curve[:, 65536-1000:65536:125])
    print("Every 10, 65436 to 65536")
    print(gamma_curve[:, 65536-100:65536:10])
    print("Every 1, 65526 to 65536")
    print(gamma_curve[:, 65536-10:65536:1])
    return gamma_curve

gamma_curve = generate_gamma_curve(gammas)

g_loss = np.zeros((5000, 1))#Why not put 6000 here too like above?

allfolders = glob.glob(result_dir + '*0')#why is there a zero at the end? save_freq could be 1 before.
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))
print("last epoch of previous run: " + str(lastepoch))

#Load all images so that GPU doesn't wait
st = time.time()
cnt = 0
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
    #print(np.amax(input_image))
    #print(np.amin(input_image))

    assigned_image_gamma = assigned_image_gammas[ind]#gamma for the image
    #print("Gamma: " + str(assigned_image_gamma))
    assigned_image_gamma_index = gammas.index(assigned_image_gamma)#Should be same as in assigned_image_gamma_indices
    #out_image = gamma_curve[assigned_image_gamma_index][out_image]#Is it int?
    
    #https://en.wikipedia.org/wiki/Rec._709#Transfer_characteristics - automatic gamma = (2.222, 4.5)?
    #https://letmaik.github.io/rawpy/api/rawpy.Params.html?highlight=gamma
    #http://goldsequence.blogspot.com/2019/11/libraw-post-processing-pipeline.html
    #im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    #gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)
    input_image = np.multiply(input_image, 65535)#CAN I DO THIS? Will it remove  the information on Gamma?
    input_image = input_image.astype(int)
    input_image = np.array(gamma_curve)[assigned_image_gamma_index][input_image]
    #IndexError: arrays used as indices must be of integer (or boolean) type
    
    #LOOKS LIKE INPUT IS multiplied by 65536
    input_image = np.multiply(input_image, 1./65535.)
    input_images[ind] = input_image
    #input_image_copy = np.copy(input_image)
    #np.squeeze(input_image_copy)
    cnt += 1
    #if(ind ==0):
    print("min, max, mean, gamma, argmax: %.5f, %.5f, %.5f, %.5f, %d" % (np.min(input_image), np.max(input_image), np.mean(input_image), assigned_image_gamma, np.argmax(input_image)))
print("%d images loaded to CPU RAM in Time=%.3f seconds." % (cnt, time.time() - st))

input_images = np.array(input_images)
print(input_images.shape)
print("\nmoved images data to numpy array")
#train_dataset = tf.data.Dataset.from_tensor_slices((input_images, assigned_image_gammas))
#train_dataset = train_dataset.repeat(epoch - lastepoch)
BATCH_SIZE = 16
no_of_batches = len(train_ids)//BATCH_SIZE
#train_dataset = train_dataset.batch(BATCH_SIZE)
#iterator = train_dataset.make_one_shot_iterator()
moving_loss = 0
moving_loss_alpha = 0.99

init_time = time.time()
learning_rate = 1e-4
input_patch_list = [None] * BATCH_SIZE
for epoch in range(lastepoch, 4001):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    cnt = 0
    if epoch > 2000:
        learning_rate = 1e-5

    epoch_init_time= time.time()
    random_iter = np.random.randint(0, len(train_ids), no_of_batches * BATCH_SIZE)
    random_iter = random_iter.reshape((no_of_batches, BATCH_SIZE))
    for i in range(no_of_batches):
        ind = random_iter[i]
        #ind = np.array(train_ids)[ind]
        if(DEBUG == 1 and epoch == lastepoch and cnt == 0):
            print("Starting Training on index " + str(ind) + ", dataset index: " + str(np.array(train_ids)[ind]))
        assigned_image_gamma = assigned_image_gammas[ind]#gamma for the image
        if(DEBUG == 1 and epoch == lastepoch and cnt == 0):
            print("Starting Training on gammas " + str(assigned_image_gamma))
        #print("Gamma: " + str(assigned_image_gamma))
        assigned_image_gamma_index = assigned_image_gamma_indices[ind]#Should be same as in assigned_image_gamma_indices
        #out_image = gamma_curve[assigned_image_gamma_index][out_image]#Is it int?
        
        st = time.time()
        cnt += 1

        # crop
        H = input_images[ind[0]].shape[0]#index could be just 0 instead of ind[0]
        W = input_images[ind[0]].shape[1]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        ##LOOKS LIKE INPUT IS multiplied by 65536!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!XXXXXXXXXXXX
        #view = np.array(ind)
        #input_patch = input_images.view(view)#arbitraries views don't work. Setting with arbitrary views works because of python inplace evaluation
        #input_patch = input_images[ind, yy:yy + ps, xx:xx + ps, :]doesn't work
        for k in range(BATCH_SIZE):
            input_patch_list[k] = input_images[ind[k], yy:yy + ps, xx:xx + ps, :]
        input_patch = np.array(input_patch_list)


        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)

        #assigned_image_gamma = np.array([assigned_image_gamma])#conversion for the sake of gt_gamma
        #Could've used np.full() for this
        #assigned_image_gamma = np.expand_dims(assigned_image_gamma, axis=0)
        #print(tf.shape(input_patch))
        #assigned_image_gamma_index = np.array([assigned_image_gamma_index])#conversion for the sake of gt_gamma
        assigned_image_gamma_index = np.array([assigned_image_gamma_index])#conversion for the sake of gt_gamma
        #assigned_image_gamma_index = np.expand_dims(assigned_image_gamma_index, axis = 0)#batch size = 1
        #ValueError: Cannot feed value of shape (1,) for Tensor u'Placeholder_1:0', which has shape '(?, 1)'
        
        #_, G_current, output = sess.run([G_opt, G_loss, out_gamma],
        #                                feed_dict={in_image: input_patch, gt_gamma: assigned_image_gamma, lr: learning_rate})
        _, G_current, output = sess.run([G_opt, G_loss, out_gamma],
                                        feed_dict={in_image: input_patch, gt_gamma: assigned_image_gamma_index, lr: learning_rate})
        #assigned_image_gamma_index is wrong here. mistake
        #TypeError: unhashable type: 'numpy.ndarray'
        #Due to in_image in the loop having the same name as the placeholder variable
        #print(tf.shape(output))
        #print(tf.shape(G_current))
        output = np.minimum(np.maximum(output, 0.0001), 1000)#bounds for gamma
        #ValueError: can only convert an array of size 1 to a Python scalar
        g_loss[ind] = G_current

        #moving_loss = moving_loss_alpha*moving_loss + (1 - moving_loss_alpha)*np.mean(g_loss[np.where(g_loss)]).item()
        if(cnt == 1):
            training_dataset_loss = np.mean(g_loss[np.where(g_loss)])
            print("Epoch %d: at batch %d: Loss=%.3f, Time=%.3f" % (
                epoch, cnt, training_dataset_loss, time.time() - st))

        if epoch % save_freq == 0:
            #file_path = result_dir + 'intermediate_results.txt'
            #text_file = open(file_path, 'a')
            #output = output.item()
            #G_current = G_current.item()
            #print(tf.shape(assigned_image_gamma_index))#Tensor("Shape:0", shape=(0,), dtype=int32)
            #n = text_file.write('Epoch %04d\t%05d:\t%d\t%.3f\t%.3f\n' % (
            #    epoch, train_id, assigned_image_gamma_index, output, G_current))
            #TypeError: %d format: a number is required, not numpy.ndarray
            #text_file.close()
            #Save gamma results in a txt file????
            #temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            #scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
            #    result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.mkdir(result_dir + '%04d' % epoch)
            
    if((epoch - lastepoch) % 10 == 1):
        print("Epoch %d: Time=%.3f, Epoch time = %.3f, Avg epoch time=%.3f\n" % (
            epoch, time.time() - init_time, time.time() - epoch_init_time, 
            (time.time() - init_time) // (epoch - lastepoch + 1)))
    if((epoch - lastepoch) % 10 == 1):
        print(g_loss)
    saver.save(sess, checkpoint_dir + 'model.ckpt')
