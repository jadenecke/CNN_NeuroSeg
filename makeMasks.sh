#! /bin/bash

source /opt/minc/1.9.18/minc-toolkit-config.sh

# Make union mask:
echo "Make union mask..."
mincmath -or -clobber input/labels/*.mnc input/union.mnc
python3 mincBloat.py -in input/union.mnc -out input/unionExpandedBy3.mnc -size 3

#Make positive mask
echo "Make positive mask..."
python3 mincPosMask.py -in_list trainLabels.txt -out input/posMask.mnc

