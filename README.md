Code and data used in publication [Accurate and robust segmentation of neuroanatomy in T1-weighted MRI by combining spatial priors with deep convolutional neural networks](https://doi.org/10.1002/hbm.24803)

## Prerequisites

* Tested on Python 3.6.3
* [MINC toolkit](https://bic-mni.github.io/)
* [Pyminc](https://github.com/Mouse-Imaging-Centre/pyminc)
* [Lasagne + Theano](https://lasagne.readthedocs.io/en/latest/)
* [Scipy](https://www.scipy.org/install.html)

## Example code

Train a network:

```
python train_cnn.py -tr_ima yourTrainingImages.txt -tr_mask yourTrainingMasks.txt -model_name yourModelName -sampling_mask yourSamplingMask.mnc -num_labels yourLabelNumber

```

Apply it:

```
python apply_cnn.py -te_ima yourTrainingImages.txt -output_dir yourOutputDir -model_name yourModelName -sampling_mask yourSamplingMask.mnc -num_labels yourLabelNumber -positive_mask yourPositiveMask.mnc

```

## Details
Images should be intensity-normalized and registered in a common space. Make sure your label images are integer-valued (background should have a value of 0). See paper and code comments for further details.


#adaption by jadenecke:
Added scripts and shell wrapper to create masks. Shell wrapper expects a input/brains/*.mnc input/labels/*.mnc structure, similar to the MAGeT algorithm
Forking: https://gist.github.com/jagregory/710671
