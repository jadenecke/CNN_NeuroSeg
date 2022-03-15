import argparse
parser = argparse.ArgumentParser(description='Bloat/expand label by a certain size.')
parser.add_argument('-in_list', action="store", type=str, dest="image_text_list", help="Path to text file specifying input images.",metavar='')
parser.add_argument('-out', action="store", type=str, dest="dest", help="Path to output image.",metavar='')
args = parser.parse_args()



import numpy as np 
import scipy
import library


#read images:
image_list = library.read_text_as_list(args.image_text_list); 
images = library.read_list_into_array(image_list)

mask = images[0,0] == images[1,0]
for i in range(len(image_list)):
	mask = (images[0,0] == images[i,0]) & (mask)


outvol = images[0].copy()
outvol = np.where(mask, images[0,0], 0)

library.write_minc_image(outvol, image_list[0], args.dest)
