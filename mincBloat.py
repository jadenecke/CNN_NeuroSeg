import argparse
parser = argparse.ArgumentParser(description='Bloat/expand label by a certain size.')
parser.add_argument('-in', action="store", type=str, dest="source", help="Path to input image.",metavar='')
parser.add_argument('-out', action="store", type=str, dest="dest", help="Path to output image.",metavar='')
parser.add_argument('-size', action="store", type=int, dest="expVal", help="Expansion Size",metavar='')
args = parser.parse_args()

import numpy as np 
import scipy
import library



invol = library.read_minc_image(args.source)

outvol = invol.copy()
k = np.ones(np.repeat(args.expVal*2+1, outvol[0].ndim))


outvol[0] = scipy.ndimage.convolve(outvol[0], k)
outvol = np.where(outvol > 0, 1 , 0)

library.write_minc_image(outvol, args.source, args.dest)
