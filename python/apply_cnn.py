import argparse

parser = argparse.ArgumentParser(description='Apply trained CNNs.')
parser.add_argument('-te_ima', action="store", type=str, dest="image_text_list", help="Path to text file specifying training images.",metavar='')
parser.add_argument('-output_dir', action="store", type=str, dest="output_dir", help="Path to folder where the output segmentations are.",metavar='')
parser.add_argument('-model_name', action="store", type=str, dest="model_name", help="Path to trained network.",metavar='')
parser.add_argument('-location', default=1, action="store", type=int, dest="location", help="Use location?",metavar='')
parser.add_argument('-sampling_mask', action="store", type=str, dest="sampling_mask", help="Path to sampling mask.",metavar='')
parser.add_argument('-positive_mask', action="store", type=str, dest="positive_mask", help="Path to positive mask.",metavar='')
parser.add_argument('-sampling_masks', action="store", type=str, dest="sampling_mask_text_list", help="Path to text file specifying sampling mask images - don't need if single sampling mask is specified.",metavar='')
parser.add_argument('-num_labels', action="store", type=int, dest="num_labels", help="Number of labels",metavar='')
parser.add_argument('-input_patch_radius', default=20, action="store", type=int, dest="input_patch_radius", help="Input patch radius",metavar='')
parser.add_argument('-output_patch_radius', default=4, action="store", type=int, dest="output_patch_radius", help="Output patch radius",metavar='')
args = parser.parse_args()

import time
import numpy as np
import pyminc.volumes.factory as pyminc
import library
from network_definitions import fc
import lasagne
import theano
import theano.tensor as T
import math
import os
from scipy import ndimage
import random

batch_size = 32

num_labels = args.num_labels
output_patch_radius = args.output_patch_radius
output_patch_width = 2*args.output_patch_radius + 1
stride = output_patch_width

input_patch_radius = args.input_patch_radius        
patch_width = 2*args.input_patch_radius + 1
output_dir = args.output_dir

list = library.read_text_as_list(args.image_text_list); num_modalities = np.size(list[0].split(','))
splitted = list[0].split(','); 

try:
    sampling_mask_list = library.read_text_as_list(args.sampling_mask_text_list)
except:
    sampling_mask_list = library.read_text_as_list(args.image_text_list); 
    for i in range(0,np.size(sampling_mask_list)):
        sampling_mask_list[i] = args.sampling_mask

# make directory if it doesn't exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

patch_var = T.tensor5('patch')
loc_var = T.tensor5('loc')
target_var = T.imatrix('targets')
learning_rate = T.scalar(name='learning_rate')

# Get network
[network,shaped] = fc(args.input_patch_radius, args.output_patch_radius, patch_var, loc_var, target_var, args.location, num_modalities, num_labels)

with np.load(args.model_name) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

prediction = lasagne.layers.get_output(shaped,deterministic=True)
predict_function = theano.function([patch_var,loc_var],prediction,on_unused_input='ignore')

# testing info
testing_image_list = library.read_text_as_list(args.image_text_list) 

# make location image
image = library.read_minc_image(testing_image_list[0])
image = library.pad_volume(image,args.input_patch_radius)
location_ima = library.make_location_ima(np.squeeze(image[0,:,:,:]))

for subj in range(0,np.size(testing_image_list,0)):

    start = time.time()

    # read in images
    image = library.read_minc_image(testing_image_list[subj])
    mask_to_do = library.read_minc_image(sampling_mask_list[subj])
    positive = library.read_minc_image(args.positive_mask)

    # pad images
    image = library.pad_volume(image,args.input_patch_radius); 
    mask_to_do = library.pad_volume(mask_to_do,args.input_patch_radius)
    positive = library.pad_volume(positive,args.input_patch_radius)
    mask = positive

    # make grid on which to evaluate patches
    grid = np.zeros(np.shape(np.squeeze(mask_to_do)))
    grid[0::stride,0::stride,0::stride] = 1; 
    mask_to_do_dilated = ndimage.morphology.binary_dilation(np.squeeze(mask_to_do).astype(np.int),structure=np.ones((stride,stride,stride)).astype(np.int))
    array_of_coords = np.transpose(np.nonzero(mask_to_do_dilated*grid)) 

    print ('Segmenting...')

    d = 0

    seg = np.zeros([num_labels,np.size(image,1),np.size(image,2),np.size(image,3)])

    # now process in batches
    num_batches = np.int(math.ceil(np.size(array_of_coords,0) / np.float32(batch_size)))
    predictions = np.zeros([np.size(array_of_coords,0),num_labels,output_patch_width,output_patch_width,output_patch_width])

    for bn in range(num_batches):
        lower = batch_size*bn
        upper = min(batch_size*(bn+1),np.size(array_of_coords,0))

        x_test, l_test, y_test = library.get_training_data_ima(image,mask,location_ima,array_of_coords[lower:upper,:],args.input_patch_radius,args.output_patch_radius)
        x_test = x_test.astype(np.float32); l_test = l_test.astype(np.float32) 
        predictions[lower:upper,:,:,:,:] += predict_function(x_test,l_test)

    prob = np.zeros([num_labels,np.size(mask,1),np.size(mask,2),np.size(mask,3)])
    norm = np.zeros([num_labels,np.size(mask,1),np.size(mask,2),np.size(mask,3)])

    # now fill in the image with the predictions
    for i in range(0,np.size(array_of_coords,0)):
        x = array_of_coords[i,0]
        y = array_of_coords[i,1]
        z = array_of_coords[i,2]
        prob[:,x-args.output_patch_radius:x+args.output_patch_radius+1,y-args.output_patch_radius:y+args.output_patch_radius+1,z-args.output_patch_radius:z+args.output_patch_radius+1] += np.squeeze(predictions[i,:,:,:,:])
        norm[:,x-args.output_patch_radius:x+args.output_patch_radius+1,y-args.output_patch_radius:y+args.output_patch_radius+1,z-args.output_patch_radius:z+args.output_patch_radius+1] += np.ones([num_labels,output_patch_width,output_patch_width,output_patch_width])

    seg[:,:,:,:] = np.nan_to_num(prob/norm)

    # get hard segmentation
    av_seg = np.argmax(seg,axis=0)
    av_seg = (av_seg * np.squeeze(mask_to_do))

    # add positive mask and unpad
    av_seg = av_seg + np.squeeze(positive)
    av_seg = np.squeeze(library.unpad_volume(av_seg,args.input_patch_radius))

    image_filename = splitted[0]
    in_vol = pyminc.volumeFromFile(image_filename)

    name = testing_image_list[subj].rsplit('/',1)[-1] # this will take the name as everything after the last forward slash
    out_vol = pyminc.volumeFromInstance(in_vol,output_dir + name)
    print("Done: " + output_dir + name)
    out_vol.data = np.round(av_seg)
    out_vol.writeFile()

    in_vol.closeVolume()
    out_vol.closeVolume()
