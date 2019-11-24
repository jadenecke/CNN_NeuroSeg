import numpy as np
import math
import random
import lasagne
import scipy.io
from scipy.ndimage.filters import gaussian_filter
import library

def train_network(network,train_fn,val_fn,args):

    input_patch_radius = args.input_patch_radius
    output_patch_radius = args.output_patch_radius

    # read in lists
    image_list = library.read_text_as_list(args.image_text_list); 
    mask_list = library.read_text_as_list(args.mask_text_list); 

    try:
        sampling_mask_list = library.read_text_as_list(args.sampling_mask_text_list)
    except:
        sampling_mask_list = library.read_text_as_list(args.image_text_list); 
        for i in range(0,np.size(image_list)):
            sampling_mask_list[i] = args.sampling_mask

    shuffler = [i for i in range(np.size(image_list))]; np.random.shuffle(shuffler)
    image_list = image_list[shuffler]; mask_list = mask_list[shuffler]; sampling_mask_list = sampling_mask_list[shuffler];

    # split into training/validation
    val_inds = [i for i in range(np.ceil(np.size(image_list)/5.0).astype(np.int32))]
    val_image_list = image_list[val_inds]; train_image_list = np.delete(image_list,val_inds)
    val_mask_list = mask_list[val_inds]; train_mask_list = np.delete(mask_list,val_inds)
    val_sampling_mask_list = sampling_mask_list[val_inds]; train_sampling_mask_list = np.delete(sampling_mask_list,val_inds)

    # read validation data into arrays
    print ("Loading images...")
    val_images, val_sampling_masks, val_masks = library.read_lists_into_arrays(val_image_list,val_mask_list,val_sampling_mask_list)
    
    # read training data into arrays
    train_images, train_sampling_masks, train_masks = library.read_lists_into_arrays(train_image_list,train_mask_list,train_sampling_mask_list)
    print ("Done.")

    # pad all
    val_images, val_sampling_masks, val_masks = library.pad_arrays(val_images,val_masks,val_sampling_masks,args.input_patch_radius)
    train_images, train_sampling_masks, train_masks = library.pad_arrays(train_images,train_masks,train_sampling_masks,args.input_patch_radius)

    # use ~N_val validation patches and ~N_train training patches per iteration
    nppi_val = np.round(args.N_val/np.size(val_images,0))
    nppi_train = np.round(args.N_train/np.size(train_images,0))

    print ("Extracting validation samples...")
    # get validation patches and remove validation images from memory
    x_val, l_val, y_val = library.get_training_data_preloaded(val_images,val_masks,val_sampling_masks,nppi_val,1,input_patch_radius,output_patch_radius)
    del val_images; del val_masks; del val_sampling_masks
    y_val = library.split_multi_label_image(y_val,args.num_labels)
    print ("Done.")     

    # training parameters
    best_val_err = 1e6 # some big number
    d = 0
    iter = 1
    num_patches_per_minibatch = 32 
    lr = args.lr
    patience = 30

    imo = args.input_patch_radius - args.output_patch_radius

    old_state = random.getstate()
    random.seed()
    randstr = str(random.randint(0,1e6))
    random.setstate(old_state)
    
    print("Starting training!")
    while (d < patience):
            
            e = 0

            if (args.aug1 != 0 or args.aug2 != 0):
                output_patch_radius = input_patch_radius             

            x_train, l_train, y_train = library.get_training_data_preloaded(train_images,train_masks,train_sampling_masks,nppi_train,0,input_patch_radius,output_patch_radius)
            y_train = library.split_multi_label_image(y_train,args.num_labels)

            if (args.aug1 != 0 or args.aug2 !=0):
                x_train, l_train, y_train = library.augment_patches(x_train, l_train, y_train, args.aug1, args.aug2, args.num_labels)
                y_train = y_train[:,:,imo:-imo,imo:-imo,imo:-imo]

            shuffler = [i for i in range(np.size(x_train,0))]; np.random.shuffle(shuffler)
            x_train = x_train[shuffler,:,:,:,:]; l_train = l_train[shuffler,:,:,:,:]; y_train = y_train[shuffler,:,:,:,:]

            print ("Iteration " + str(iter))

            train_err = 0; train_batches = 0
            val_err = 0; val_acc = 0; val_batches = 0

            # training
            num_batches = np.int(math.floor(np.size(x_train,0) / num_patches_per_minibatch))

            for bn in range(num_batches):

                lower = num_patches_per_minibatch*bn
                upper = num_patches_per_minibatch*(bn+1)
                x_mb = x_train[lower:upper,:,:,:,:]  
                l_mb = l_train[lower:upper,:,:,:,:]
                y_mb = y_train[lower:upper,:,:,:,:]

                y_mb = np.transpose(y_mb, (1,0,2,3,4))
                y_mb = np.reshape(y_mb,[np.size(y_mb,0),np.size(y_mb,1)*np.size(y_mb,2)*np.size(y_mb,3)*np.size(y_mb,4)])
                y_mb = np.transpose(y_mb,(1,0)).astype(np.float32)

                train_err += train_fn(x_mb,l_mb,y_mb,lr)
                train_batches += 1

            # validating
            num_batches_v = np.int(math.floor(np.size(x_val,0) / num_patches_per_minibatch));
                    
            for bn_v in range(num_batches_v):

                lower = num_patches_per_minibatch*(bn_v)
                upper = num_patches_per_minibatch*(bn_v+1)
                x_mb = x_val[lower:upper,:,:,:,:]
                l_mb = l_val[lower:upper,:,:,:,:]
                y_mb = y_val[lower:upper,:,:,:,:]
                
                y_mb = np.transpose(y_mb,(1,0,2,3,4))
                y_mb = np.reshape(y_mb,[np.size(y_mb,0),np.size(y_mb,1)*np.size(y_mb,2)*np.size(y_mb,3)*np.size(y_mb,4)])
                y_mb = np.transpose(y_mb,(1,0)).astype(np.float32)                                  
                
                err, acc = val_fn(x_mb,l_mb,y_mb)
                val_err += err
                val_acc += acc
                val_batches += 1

            print("Validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("Training loss:\t\t{:.6f}".format(train_err / train_batches))

            if (val_err / val_batches) < best_val_err:
                best_val_err = (val_err / val_batches)
                best_params = lasagne.layers.get_all_param_values(network)
                d = 0
            else:
                d = d + 1

            iter = iter + 1

    print("Done training!")
    lasagne.layers.set_all_param_values(network, best_params)

    return network

