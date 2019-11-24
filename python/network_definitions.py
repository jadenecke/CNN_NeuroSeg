import lasagne
import theano
import theano.tensor as T
import random
import math
import numpy as np
from lasagne.layers import batch_norm, ReshapeLayer
from lasagne.init import Normal

def fc(input_patch_radius, output_patch_radius, patch_var, loc_var, target_var, location, num_modalities, num_labels):

    input_patch_width = 2*input_patch_radius + 1
    output_patch_width = 2*output_patch_radius + 1
    num_conv_layers = input_patch_radius - output_patch_radius

    # Network architecture
    c = lasagne.layers.InputLayer(shape=(None,num_modalities,input_patch_width,input_patch_width,input_patch_width), input_var=patch_var)
    if (num_modalities == 1):
       c = lasagne.layers.ExpressionLayer(c, lambda X: T.addbroadcast(X, 1), lambda s: s)

    if (location == 1):
        loc = lasagne.layers.InputLayer(shape=(None,3,input_patch_width,input_patch_width,input_patch_width), input_var=loc_var)
        c = lasagne.layers.ConcatLayer([c, loc], axis=1)

    c = lasagne.layers.BatchNormLayer(c)

    for layer in range(1,num_conv_layers+1):
        c = lasagne.layers.Conv3DLayer(c, num_filters=32, filter_size=(3,3,3), nonlinearity=lasagne.nonlinearities.elu)
        if (layer == 1):
           c2 = c
        if (layer > 1):
           c2 = lasagne.layers.ConcatLayer([c2,c], axis=1, cropping=(None,None,'center','center','center'))

    c = lasagne.layers.BatchNormLayer(c2)

    c = lasagne.layers.Conv3DLayer(c, num_filters=128, filter_size=(1,1,1), nonlinearity=lasagne.nonlinearities.elu)
    c = lasagne.layers.DropoutLayer(c, p=0.1)
    c = lasagne.layers.Conv3DLayer(c, num_filters=64, filter_size=(1,1,1), nonlinearity=lasagne.nonlinearities.elu)
    c = lasagne.layers.DropoutLayer(c, p=0.1)
    c = lasagne.layers.Conv3DLayer(c, num_filters=num_labels, filter_size=(1,1,1), nonlinearity=None)
  
    # Spatial softmax
    c = lasagne.layers.DimshuffleLayer(c,(1,0,2,3,4))
    c = lasagne.layers.FlattenLayer(c, outdim=2)
    c = lasagne.layers.DimshuffleLayer(c,(1,0))
    network = lasagne.layers.NonlinearityLayer(c, nonlinearity=lasagne.nonlinearities.softmax)

    shaped = lasagne.layers.DimshuffleLayer(network,(1,0))

    target_shape = list(c.input_layer.input_layer.output_shape);
    target_shape[1] = -1
    target_shape = tuple(target_shape)

    shaped = lasagne.layers.ReshapeLayer(shaped, target_shape)
    shaped = lasagne.layers.DimshuffleLayer(shaped,(1,0,2,3,4))

    return (network, shaped);


