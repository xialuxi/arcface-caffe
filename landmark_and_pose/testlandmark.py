#coding=utf-8
import sys
sys.path.insert(0,'./caffe-master/python')
import caffe
import cv2
import numpy as np
import os
import json
import copy

def get_net():
    caffe.set_device(0)
    caffe.set_mode_gpu()
    caffemodel = './model/mobilenet_v2_iter_50000.caffemodel'
    deploy = './model/MobileNet-V2_deploy.prototxt'
    net = caffe.Net(deploy, caffemodel, caffe.TEST)
    return net

def get_result(net, img):

    half_width = img.shape[1] * 0.5
    half_height = img.shape[0] * 0.5

    tempimg = np.zeros((1, 48, 48, 3))
    scale_img = cv2.resize(img,(48,48))
    scale_img = (scale_img - 127.5) / 125.0
    tempimg[0, :, :, :] = scale_img

    tempimg = tempimg.transpose(0, 3, 1, 2)
    net.blobs['data'].data[...] = tempimg

    net.forward()
    points = copy.deepcopy(net.blobs['fc2'].data[0])
    pose = copy.deepcopy(net.blobs['fc3'].data[0])

    facelandmarks = []
    for i in range(21):
        x = points[i * 2 + 0] * half_width + half_width
        y = points[i * 2 + 1] * half_height + half_height
        point = []
        point.append(int(x))
        point.append(int(y))
        facelandmarks.append(point)

    pose = pose / np.pi * 180

    return facelandmarks, pose


def showresult(img, facelandmarks):
    for point in facelandmarks:
        cv2.circle(img, (point[0], point[1]),0,(0,0,255),2)
    return img

if __name__ == '__main__':

    net = get_net()

    imgpath = './zhiwen.jpeg'
    img = cv2.imread(imgpath)

    facelandmarks, pose = get_result(net, img)

    print 'pose: ', pose

    show = showresult(img, facelandmarks)

    cv2.imshow('', show)
    cv2.waitKey(0)


