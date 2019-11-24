import numpy as np
import pyminc.volumes.factory as pyminc
import random
from random import randint, uniform
import os
import subprocess
from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
from scipy.ndimage.interpolation import rotate, zoom
import scipy.misc
from random import uniform
import time

def read_text_as_list(text_file):
    
    with open(text_file) as f: text_list = f.read().splitlines(); text_list = np.array(text_list) 
    
    return text_list

def extract_patch(vol,patch_center,patch_radius):
    
    num_modalities = np.size(vol,0)
    patch = np.zeros([num_modalities,2*patch_radius+1,2*patch_radius+1,2*patch_radius+1])
    for i in range(0,num_modalities):
        patch[i,:,:,:] = vol[i,patch_center[0]-patch_radius:patch_center[0]+patch_radius+1,patch_center[1]-patch_radius:patch_center[1]+patch_radius+1,patch_center[2]-patch_radius:patch_center[2]+patch_radius+1]   
        
    return patch
    
def read_minc_image(minc_filename):
    
    split = minc_filename.split(',')
    minc_vol = pyminc.volumeFromFile(split[0]);
    vol = minc_vol.data;
    vol = np.expand_dims(vol,axis=0)
    
    if (np.size(split) > 1):
        for i in range(1,np.size(split)):
            minc_vol = pyminc.volumeFromFile(split[i]);
            vol = np.concatenate((vol,np.expand_dims(minc_vol.data,axis=0)),axis=0)
    return vol

def pad_volume(vol,pad_value):
    
    vol = np.pad(vol, [(0,0),(pad_value,pad_value),(pad_value,pad_value),(pad_value,pad_value)],'constant',constant_values=0)
    return vol

def unpad_volume(vol,pad_value):
    
    vol = vol[pad_value:-pad_value,pad_value:-pad_value,pad_value:-pad_value]
    return vol    

def standardize(patch,eps):
    
    for i in range(0,np.size(patch,0)):
        patch[i,:,:,:] = (patch[i,:,:,:] - np.mean(patch[i,:,:,:])) / (np.std(patch[i,:,:,:]) + eps)
    return patch  

def get_training_data_ima(image,mask,location_ima,array_of_coords,input_patch_radius,output_patch_radius):  
    input_patch_width = 2*input_patch_radius + 1
    output_patch_width = 2*output_patch_radius + 1
    num_modalities = np.size(image,0)

    # make arrays
    x_train = np.zeros([np.size(array_of_coords,0),num_modalities,input_patch_width,input_patch_width,input_patch_width])
    y_train = np.zeros([np.size(array_of_coords,0),1,output_patch_width,output_patch_width,output_patch_width])
    l_train = np.zeros([np.size(array_of_coords,0),3,input_patch_width,input_patch_width,input_patch_width])

    # extract features
    for n in range(0,np.size(array_of_coords,0)):
        patch = extract_patch(image,array_of_coords[n],input_patch_radius); patch = standardize(patch,0.0001)
        patch_loc = extract_patch(location_ima,array_of_coords[n],input_patch_radius)
        true = extract_patch(mask,array_of_coords[n],output_patch_radius)
        
        x_train[n,:,:,:,:] = patch
        l_train[n,:,:,:,:] = patch_loc
        y_train[n,:,:,:,:] = true

    return (x_train, l_train, y_train)


def get_coords_stride(array_of_coords,stride):
 
    for i in range(np.size(array_of_coords,0)-1,-1,-1):
        if (array_of_coords[i][0] % stride != 0):
            array_of_coords = np.delete(array_of_coords,i,axis=0)

    for i in range(np.size(array_of_coords,0)-1,-1,-1):
        if (array_of_coords[i][1] % stride != 0):
            array_of_coords = np.delete(array_of_coords,i,axis=0)

    for i in range(np.size(array_of_coords,0)-1,-1,-1):
        if (array_of_coords[i][2] % stride != 0):
            array_of_coords = np.delete(array_of_coords,i,axis=0)

    return array_of_coords    
       
def get_training_data_preloaded(images,masks,sampling_masks,max_patches_per_image,uniform_sampling,input_patch_radius,output_patch_radius):

    eps = 0.0001
    location_ima = make_location_ima(np.squeeze(images[0,0,:,:,:]))
    num_labels = np.amax(masks[0,0,:,:,:]).astype(int) + 1

    for subj in range(0,np.size(images,0)):

        image = images[subj,:,:,:,:]; 
        mask = masks[subj,:,:,:,:]
        sampling_mask = sampling_masks[subj,:,:,:,:]

        if (uniform_sampling == 1):
            array_of_coords = np.transpose(np.nonzero(np.squeeze(sampling_mask)))
            np.random.shuffle(array_of_coords) # shuffle rows (coords)
            array_of_coords = array_of_coords[0:int(max_patches_per_image,):]
  
        if (uniform_sampling == 0):
            for label in range(0,num_labels):
                label_mask = np.zeros(np.shape(mask)); label_mask[mask == label] = 1;
                array_of_coords_label = np.transpose(np.nonzero(np.multiply(np.squeeze(sampling_mask),np.squeeze(label_mask))))
                np.random.shuffle(array_of_coords_label) # shuffle rows (coords)
                array_of_coords_label = array_of_coords_label[0:int(np.round(max_patches_per_image/num_labels)),:]

                if label == 0:
                    array_of_coords = array_of_coords_label
                else:
                    array_of_coords = np.concatenate((array_of_coords,array_of_coords_label),axis=0)

        x_train_ima, l_train_ima, y_train_ima = get_training_data_ima(image,mask,location_ima,array_of_coords,input_patch_radius,output_patch_radius) 
        
        if subj == 0:
            x_train = x_train_ima; 
            l_train = l_train_ima; 
            y_train = y_train_ima;
        else:
            x_train = np.concatenate((x_train,x_train_ima),axis=0); 
            l_train = np.concatenate((l_train,l_train_ima),axis=0); 
            y_train = np.concatenate((y_train,y_train_ima),axis=0); 

    # convert to float
    x_train = x_train.astype(np.float32); l_train = l_train.astype(np.float32); y_train = y_train.astype(np.int32)

    print ("Extracted " + str(np.size(x_train,0)) + " patches.")

    return (x_train, l_train, y_train)

def read_list_into_array(list):
    
    array = read_minc_image(list[0]); array = np.expand_dims(array,axis=0)
    for n in range(1,np.size(list)):
        d = read_minc_image(list[n]); d = np.expand_dims(d,axis=0); array = np.concatenate((array,d),axis=0)
    return array

def read_lists_into_arrays(image_list,mask_list,sampling_mask_list):
    
    images = read_list_into_array(image_list)
    masks = read_list_into_array(mask_list)
    sampling_masks = read_list_into_array(sampling_mask_list)
    
    return images, sampling_masks, masks

def pad_arrays(images,masks,sampling_masks,pad_value):
    
    images = np.pad(images, ((0,0),(0,0),(pad_value,pad_value),(pad_value,pad_value),(pad_value,pad_value)),'constant',constant_values=0)
    masks = np.pad(masks, ((0,0),(0,0),(pad_value,pad_value),(pad_value,pad_value),(pad_value,pad_value)),'constant',constant_values=0)
    sampling_masks = np.pad(sampling_masks, ((0,0),(0,0),(pad_value,pad_value),(pad_value,pad_value),(pad_value,pad_value)),'constant',constant_values=0)
    
    return images, sampling_masks, masks

def make_location_ima(image):
    
    x_image = np.zeros(np.shape(image))
    for d in range(0,np.size(image,0)):
        x_image[d,:,:] = (d+1)*np.ones([np.size(image,1),np.size(image,2)])
    x_image = np.expand_dims(x_image,axis=0)
    x_image = x_image / np.max(x_image)    

    y_image = np.zeros(np.shape(image))
    for d in range(0,np.size(image,1)):
        y_image[:,d,:] = (d+1)*np.ones([np.size(image,0),np.size(image,2)])
    y_image = np.expand_dims(y_image,axis=0) 
    y_image = y_image / np.max(y_image)

    z_image = np.zeros(np.shape(image))
    for d in range(0,np.size(image,2)):
        z_image[:,:,d] = (d+1)*np.ones([np.size(image,0),np.size(image,1)])    
    z_image = np.expand_dims(z_image,axis=0) 
    z_image = z_image / np.max(z_image)

    location_ima = np.concatenate((x_image,y_image,z_image),axis=0)

    return location_ima

def write_minc_image(array,example_image_filename,output_name):

    in_vol = pyminc.volumeFromFile(example_image_filename)

    out_vol = pyminc.volumeFromInstance(in_vol,output_name)
    out_vol.data = np.squeeze(array);
    out_vol.writeFile()

    in_vol.closeVolume()
    out_vol.closeVolume()

def augment_patches(x_train,l_train,y_train,p1,p2,num_labels):

    num_modalities = np.size(x_train,1)
    concated = np.concatenate((x_train,l_train,y_train),axis=1)

    for d in range(0,np.size(concated,0)):
        concated[d,:,:,:,:] = deform_images(np.squeeze(concated[d,:,:,:,:]),p1,p2)

    x_train[:,0:num_modalities,:,:,:] = concated[:,0:num_modalities,:,:,:]
    l_train[:,0:3,:,:,:] = concated[:,num_modalities:num_modalities+3,:,:,:]
    y_train[:,0:num_labels,:,:,:] = concated[:,num_modalities+3:num_modalities+3+num_labels,:,:,:]
    
    return x_train, l_train, y_train

def deform_images(image,sigma,alpha): 

    deformed = np.zeros(np.shape(image))

    shape = np.squeeze(image[0,:,:,:]).shape

    random_state = np.random.RandomState(None)

    # make a random field, smooth
    dx = gaussian_filter((random_state.rand(*shape)*2 - 1), sigma)
    dy = gaussian_filter((random_state.rand(*shape)*2 - 1), sigma)
    dz = gaussian_filter((random_state.rand(*shape)*2 - 1), sigma)

    displacements = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
    mean_displacement = np.mean(displacements)

    # scale
    dx = dx/mean_displacement * alpha 
    dy = dy/mean_displacement * alpha
    dz = dz/mean_displacement * alpha
 
    # resample
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1, 1))    

    for n in range(0,np.size(image,0)):
            deformed[n,:,:,:] = map_coordinates(np.squeeze(image[n,:,:,:]), indices, order=1, mode='reflect').reshape(shape)

    return deformed

def split_multi_label_image(mask,num_labels):
    
    split = np.zeros((np.size(mask,0),num_labels,np.size(mask,2),np.size(mask,3),np.size(mask,4)))
    
    for b in range(0,np.size(mask,0)):
        split_b = np.zeros((num_labels,np.size(mask,2),np.size(mask,3),np.size(mask,4)))
        for n in range(0,num_labels):
            empty = np.zeros((np.size(mask,2),np.size(mask,3),np.size(mask,4)))
            empty[mask[b,0,:,:,:] == n] = 1;
            split_b[n,:,:,:] = empty
        split[b,:,:,:,:] = split_b

    return split
    

