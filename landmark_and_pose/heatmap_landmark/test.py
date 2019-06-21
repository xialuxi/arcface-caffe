#coding=utf-8
import sys
sys.path.insert(0, './caffe-master/python')
import caffe
import cv2
import numpy as np
import os
import copy
import math

def get_net():
    caffe.set_device(0)
    caffe.set_mode_gpu()
    caffemodel = './face_landmark_restnet18.caffemodel'
    deploy = './face_landmark_restnet18.prototxt'
    net = caffe.Net(deploy, caffemodel, caffe.TEST)
    return net


def get_heatmap(net, image):
    width = img.shape[1]
    height = img.shape[0]
    tempimg = np.zeros((1, 96, 96, 3))
    scale_img = cv2.resize(img, (96, 96))
    scale_img = (scale_img - 127.5) * 0.0078125
    tempimg[0, :, :, :] = scale_img
    tempimg = tempimg.transpose(0, 3, 1, 2)
    net.blobs['data'].data[...] = tempimg
    net.forward()
    heatmap = copy.deepcopy(net.blobs['conv25'].data[0])
    print('heatmap: ', heatmap.shape)

    heatmap = heatmap.reshape((heatmap.shape[0], -1))

    points = []
    scores = []
    for i in range(heatmap.shape[0]):
        maxind = np.argmax(heatmap[i])
        topk_ys = maxind / 48
        topk_xs = maxind % 48
        point = []
        point.append(round(topk_xs  * width / 48.0))
        point.append(round(topk_ys * height / 48.0))

        scores.append(heatmap[i][maxind])
        points.append(np.array(point, dtype=np.int32))

    return points, scores



if __name__ == '__main__':

    net = get_net()
    imgpath = './test.jpg'
    img = cv2.imread(imgpath)
    points, scores = get_heatmap(net, img)
    print 'points: ', points
    print 'scores: ', scores