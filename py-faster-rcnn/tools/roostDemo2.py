#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import xml.etree.ElementTree as ET
#sys.path.append('usr/local/lib/python2.7/dist-packages/')

CLASSES = ('__background__',
           'roost')

NETS = {'zf': ('ZF',
                  'roostModel.caffemodel')}

truePos = 0
allRoosts = 0 
allDetections = 0

def vis_detections(im, class_name, dets, im_file, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    #if len(inds) == 0:
    #   print("Nothing Detected")  
    #   return 0,0,0
    
    thefile = open('overlaps.txt','a')
    thefile.write(im_file+'\n')    
    truePos = 0
    allRoosts = 0
    allDetections = 0
    #add original labels 
    filename = os.path.join('/home/sgabriel/py-faster-rcnn/data/CNNData/Annotations',im_file[:-4] + '.xml')
    print(filename)
    tree = ET.parse(filename)
    objs = tree.findall('object')
    boxes = np.zeros((len(objs),4),dtype=np.uint16)
    if len(inds) == 0:
       print("Nothing Detected")
       plt.savefig(im_file)
       return 0,len(objs),0,0 
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
     
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)-1 
        y1 = float(bbox.find('ymin').text)-1 
        x2 = float(bbox.find('xmax').text)-1
        y2 = float(bbox.find('ymax').text)-1
        boxes[ix,:] = [x1,y1,x2,y2]
        ax.add_patch(plt.Rectangle((boxes[ix,0],boxes[ix,1]),boxes[ix,2]-boxes[ix,0],boxes[ix,3]-boxes[ix,1], fill=False, edgecolor='yellow',linewidth=3.5))
        #ax.text(boxes[ix,0],boxes[ix,1]-2,'{:s}{:.3f}'.format('ground truth',score),bbox=dict(facecolor('blue',alpha=0.5),fontsize=14,color='white')
   
    boxes_labeled = [False] * len(objs)
    allRoosts = allRoosts + len(objs)
    allDetections = allDetections + len(inds)
    actualR = len(objs)
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
           plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        ixmin = np.maximum(boxes[:,0],bbox[0])
        iymin = np.maximum(boxes[:,1],bbox[1])
        ixmax = np.minimum(boxes[:,2],bbox[2])
        iymax = np.minimum(boxes[:,3],bbox[3])
        #print(ixmin)
        #print(ixmax)
        iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
        ih = np.maximum(iymax - iymin + 1.0, 0.0)
        inters = iw * ih 
        union = ((bbox[2] - bbox[0] + 1.0) * (bbox[3] - bbox[1] + 1.0) + (boxes[:,2] - boxes[:,0] + 1.0) * (boxes[:,3] - boxes[:,1]) - inters)
        overlaps = inters / union
        ovmax = np.max(overlaps) 
        jmax = np.argmax(overlaps)
        if ovmax > .3:
           ax.text(boxes[jmax,0],boxes[jmax,1] - 2,'{:s}'.format('true positive'),bbox=dict(facecolor='blue',alpha=0.5),fontsize=14,color='white')
           truePos = truePos + 1       
        else:
           ax.text(boxes[jmax,0],boxes[jmax,1] - 2,'{:s}'.format('false negative'),bbox=dict(facecolor='blue',alpha=0.5),fontsize=14,color='white')
        boxes_labeled[jmax] = True
        #thefile.write("Detection {0}".format(i))
        thefile.write("{0}\n".format(ovmax))
    for roost in range(len(boxes)):
        if boxes_labeled[roost] == False:
           ax.text(boxes[roost,0],boxes[roost,1]-2,'{:s}'.format('false negative'),bbox=dict(facecolor='blue',alpha=0.5),fontsize=14,color='white')
           boxes_labeled[roost] = True
    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)
    thefile.close()
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    print(im_file)
    print('detections')
    print(allDetections)
    print('roosts')
    print(allRoosts)
    print('True Positives')
    print(truePos)
    #plt.savefig(im_file,bbox_inches='tight',pad_inches=0)
    return allDetections,allRoosts,truePos,actualR

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join('images',image_name)
    print("troubleshooting")
    print(im_file)
    im = cv2.imread(im_file)
    #im = plt.imread(im_file)
    im = im[...,::-1]
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.755 #.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        detections,roosts,tp,actualR = vis_detections(im, cls, dets,image_name, thresh=CONF_THRESH)
        print("DONE")
        return detections,roosts,tp,actualR
     #print('Precision: {.3f}'.format(truePos/allDetections))
     #print('Recall: {.3f}'.format(truePos/allRoosts))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [zf]',
                        choices=NETS.keys(), default='zf')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = 'test.prototxt'
    caffemodel = 'roostModel.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    #im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    #for i in xrange(2):
       # _, _= im_detect(net, im)

    im_names = [f for f in os.listdir('images') if os.path.isfile(os.path.join('images',f))] #['541.jpg']
    allRoosts = 0
    allDetections = 0
    truePos = 0
    precision = 0.0
    actualR = 0
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        print('./images/{}'.format(im_name))
        detections,roosts,tp,rid = demo(net,im_name)
        if detections == 0:
           precision = precision + 0.0
        else:
           precision = precision + 1.0
        allRoosts = allRoosts + roosts 
        allDetections = allDetections + detections 
        truePos = truePos + tp
        actualR = actualR + rid
    print(allDetections)
    print(allRoosts)
    print(truePos)
    print(precision/allDetections)
    print(actualR)
    #plt.show()
