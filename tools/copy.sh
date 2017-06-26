#!/bin/sh
cat ~/py-faster-rcnn/data/CNNData/ImageSets/Main/test.txt | while read i; do
   cp ~/py-faster-rcnn/data/CNNData/JPEGImages/$i.jpg ~/py-faster-rcnn/tools/images
done
