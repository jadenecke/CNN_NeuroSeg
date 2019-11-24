import argparse

parser = argparse.ArgumentParser(description='Train CNN.')
parser.add_argument('-tr_ima', action="store", type=str, dest="image_text_list", help="Path to text file specifying training images.",metavar='')
parser.add_argument('-tr_mask', action="store", type=str, dest="mask_text_list", help="Path to text file specifying training masks.",metavar='')
parser.add_argument('-model_name', action="store", type=str, dest="model_name", help="Name of file to write trained network.",metavar='')
parser.add_argument('-lr', default=0.0025, action="store", type=float, dest="lr", help="Learning rate",metavar='')
parser.add_argument('-location', default=1, action="store", type=int, dest="location", help="Use location information?",metavar='')
parser.add_argument('-N_train', default=1500, action="store", type=int, dest="N_train", help="Number of patches to extract per training iteration.",metavar='')
parser.add_argument('-N_val', default=2000, action="store", type=int, dest="N_val", help="Number of patches to extract for validation.",metavar='')
parser.add_argument('-sampling_mask', action="store", type=str, dest="sampling_mask", help="Path to sampling mask used for all images.",metavar='')
parser.add_argument('-sampling_masks', action="store", type=str, dest="sampling_mask_text_list", help="Path to text file specifying sampling mask images - don't need if single sampling mask is specified.",metavar='')
parser.add_argument('-aug1', default=4, action="store", type=int, dest="aug1", help="Data augmentation?",metavar='')
parser.add_argument('-aug2', default=2, action="store", type=float, dest="aug2", help="Data augmentation?",metavar='')
parser.add_argument('-num_labels', action="store", type=int, dest="num_labels", help="Number of labels",metavar='')
parser.add_argument('-input_patch_radius', default=20, action="store", type=int, dest="input_patch_radius", help="Input patch radius",metavar='')
parser.add_argument('-output_patch_radius', default=4, action="store", type=int, dest="output_patch_radius", help="Output patch radius",metavar='')
args = parser.parse_args()

# -------------------------------- Notes --------------------------------
# All images should be pre-processed before being passed into this function: at minimum, linear registration to MNI space + non-uniformity correction + linear intensity normalization

# You can use multiple modalities (or priors) by passing a text file list to -tr_ima with each line looking something like this: subj_t1.mnc,subj_t2.mnc,subj_pd.mnc
# Make sure that the order of minc files is the consistent across image/mask text lists
# Specify a single sampling mask image (used for all subjects) using -sampling_mask
# Alternatively, specify a text file list of sampling masks (one for each subject, make sure order is consistent) using -sampling_masks

# Set -location 1 if the training/testing subjects are spatially aligned
# Data augmentation: aug1 controls the smoothness (4 voxels works well), aug2 controls the magnitude (2 voxels works well) of the random elastic deformations
# Input/output radii: 20/4 work well. Note that the number of 3x3x3 convolutional layers in the network is equal to (input radius - output_radius)
# Number of labels: include background! E.g. left/right hippocampus segmentation, use -num_labels 3
# -----------------------------------------------------------------------

import lasagne
import os
import theano
import theano.tensor as T
import random
import math
import numpy as np
import random
from library import read_text_as_list, read_list_into_array, read_lists_into_arrays
from network_definitions import fc
from training_macros import train_network

# Define parameters, prepare variables
l1_penalty = 0
l2_penalty = 1e-4
list = read_text_as_list(args.image_text_list); num_modalities = np.size(list[0].split(','))

# Prepare Theano variables for inputs and targets
patch_var = T.tensor5('patch')
loc_var = T.tensor5('loc')
target_var = T.fmatrix('targets')
learning_rate = T.scalar(name='learning_rate')
iter = T.scalar(name='iter')

num_labels = args.num_labels

# Get network
[network,shaped] = fc(args.input_patch_radius, args.output_patch_radius, patch_var, loc_var, target_var, args.location, num_modalities, num_labels)   

# Create a loss expression for training
prediction = lasagne.layers.get_output(network, deterministic=False)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var); loss = loss.mean()
weightsl1 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l1)
weightsl2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
loss = loss + (l2_penalty * weightsl2) + (l1_penalty * weightsl1)

# Create update expressions for training
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.rmsprop(loss, params, learning_rate=learning_rate)
updates = lasagne.updates.apply_nesterov_momentum(updates, params, momentum=0.9)        

# Create a loss expression for validation/testing
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var); test_loss = test_loss.mean()
test_acc = T.mean(lasagne.objectives.categorical_accuracy(test_prediction, target_var))

# Compile a function performing a training step on a mini-batch (by giving the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([patch_var, loc_var, target_var, learning_rate], loss, updates=updates, on_unused_input='ignore')

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([patch_var, loc_var, target_var], [test_loss, test_acc], on_unused_input='ignore')

# train network
network = train_network(network,train_fn,val_fn,args)

# save model and validation loss curve
np.savez(args.model_name, *lasagne.layers.get_all_param_values(network))


