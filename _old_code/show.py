from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import utils

#https://stackoverflow.com/questions/35737116/runtimeerror-invalid-display-variable
plt.switch_backend('agg')#For the sake of line 47

PLOT_DIR = './out/plots'


def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    #grid_r, grid_c = utils.get_grid_dim(num_filters)
    grid_r, grid_c = 4, 8

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')




def show_what():
    w = np.array(weights)
    w = np.moveaxis(w, 2, 0)
    w = np.moveaxis(w, 3, 0)
    print(w.shape)
    total_filters_in_prev_layers = 3

    cols = 3
    rows = 1
    current_filter = 16
    fig = plt.figure(figsize=(10, 10))

    for each_depth in range(w.shape[1]):
        fig.add_subplot(rows, cols, each_depth+1)
        plt.imsave(plot_dir + "/filter_" + str(each_depth), )
        #plt.imshow(w[current_filter][each_depth], cmap='gray')
    #https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/summary/image
    #tf.summary.image('conv1', weights, max_outputs = 3)
    
    
#NOT TO BE EXECUTED
#from torchvision import utils as vutils
#import matplotlib.pyplot as plt
#def vis():
#    plot_dir = './filters'
#    if not os.path.exists(plot_dir):
#        os.makedirs(plot_dir)
#        
#    conv_weights = sess.run([tf.get_collection('g_conv1_1')])
#   #for i, c in enumerate(conv_weights[0]):
 #    #   plot_conv_weights(c, 'conv{}'.format(i))
 #   conv_weights[0]


#Visualize the weights
#def weightshow(wt):
# wt = wt * 0.3081 - 0.1307     # denormalize
#    npwt = wt.detach().numpy()
#    plt.imshow(np.transpose(npwt, (1, 2, 0)), cmap='gray')#Gray doesn't seem to work!!!
#    plt.show()
 #   for m in model.modules():#made a mistake with the name of variable 'model' when copied code
 #      #print(m)#This prints the whole model instead of one layer. Why?
  #      if isinstance(m, nn.Conv2d):
  #          print(m.weight.shape)#This prints the first layer only! Strange!!!
   ###       break # Only the first layer filters can be visualized through make_grid
    
