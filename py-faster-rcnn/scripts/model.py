import caffe,cv2 

print('end of setup')
model = '../../models/pascal_voc/ZF/faster_rcnn_end2end/solver.prototxt'
weights = '../../output/default/voc_2007_trainval/zf_faster_rcnn_iter_50.caffemodel'
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(model,weights,'test')

image = cv2.imread('540.jpg')
#scores, boxes = im_detect(net,image)
#print(scores)
#print(boxes)
