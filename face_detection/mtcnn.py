#coding=utf-8
import sys
sys.path.insert(0,'./caffe/python')
import caffe
import cv2
import os
import numpy as np
import math
import copy

std_mean = 127.5
std_scale = 0.0078125
batchsize = 128
factor = 0.709
minisize = 30

#pnet
pnet_stride = 2
pnet_cell_size = 12
pnet_thread = 0.95
#rnet
rnet_thread = 0.95
#onet
onet_thread = 0.95




def Align_sphereface(input_image, points, output_size = (96, 112)):

    image = copy.deepcopy(input_image)
    src = np.matrix([[points[0], points[2], points[4], points[6], points[8]],
                      [points[1], points[3], points[5], points[7], points[9]], [1, 1, 1, 1, 1]])
    dst = np.matrix([ [30.2946, 65.5318, 48.0252, 33.5493, 62.7299],
                      [51.6963, 51.5014, 71.7366, 92.3655, 92.2041] ])

    T = (src * src.T).I * src * dst.T
    img_affine = cv2.warpAffine(image, T.T, output_size)

    return img_affine

def Align_seqface(input_image, points, output_size = (128, 128)):
    image = copy.deepcopy(input_image)

    eye_center_x = (points[0] + points[2]) * 0.5
    eye_center_y = (points[1] + points[3]) * 0.5
    mouse_center_x = (points[6] + points[8]) * 0.5
    mouse_center_y = (points[7] + points[9]) * 0.5
    rad_tan = 1.0 * (points[3] - points[1]) / (points[2] - points[0])
    rad = math.atan(rad_tan)
    deg = np.rad2deg(rad)

    width = int(math.fabs(math.sin(rad)) * image.shape[0] + math.fabs(math.cos(rad)) * image.shape[1])
    height = int(math.fabs(math.cos(rad)) * image.shape[0] + math.fabs(math.sin(rad)) * image.shape[1])

    transformMat = cv2.getRotationMatrix2D((eye_center_x, eye_center_y), deg, 1.0)
    dst = cv2.warpAffine(image, transformMat, (width, height))

    diff_x = mouse_center_x - eye_center_x
    diff_y = mouse_center_y - eye_center_y
    r_mouse_center_y = diff_y * float(math.cos(rad)) - diff_x * float(math.sin(rad)) + eye_center_y

    d = r_mouse_center_y - eye_center_y + 1
    dx = int(d * 3 / 2.0)
    dy = int(d * 3 / 3.0)
    x0 = int(eye_center_x) - dx
    x0 = max(x0, 0)
    x1 = int(eye_center_x + (3*d - dx)) - 1
    x1 = min(x1, width-1)
    y0 = int(eye_center_y) - dy
    y0 = max(y0, 0)
    y1 = int(eye_center_y + (3*d - dy)) - 1
    y1 = min(y1, height-1)

    alignface = dst[y0:y1, x0:x1, :]
    alignface = cv2.resize(alignface, (128,128))
    return alignface

def CalScale(width, height):
    scales = []
    scale = 12.0 / minisize
    minWH = min(height, width) * scale;
    while minWH >= 12.0:
        scales.append(scale)
        minWH *= factor
        scale *= factor
    return scales

def BBoxRegression(results):
    for result in results:
        box = result['faceBox']
        bbox_reg = result['bbox_reg']
        w = box[2] - box[0] + 1
        h = box[3] - box[1] + 1
        box[0] += bbox_reg[0] * w
        box[1] += bbox_reg[1] * h
        box[2] += bbox_reg[2] * w
        box[3] += bbox_reg[3] * h
    return results

def BBoxPad(results, width, height):
    for result in results:
        box = result['faceBox']
        box[0] = round(max(box[0], 0.0))
        box[1] = round(max(box[1], 0.0))
        box[2] = round(min(box[2], width - 1.0))
        box[3] = round(min(box[3], height - 1.0))
    return results

def BBoxPadSquare(results, width, height):
    for result in results:
        box = result['faceBox']
        w = box[2] - box[0] + 1;
        h = box[3] - box[1] + 1;
        side = max(w, h)
        box[0] = round(max(box[0] + (w - side) * 0.5, 0))
        box[1] = round(max(box[1] + (h - side) * 0.5, 0.))
        box[2] = round(min(box[0] + side - 1.0, width - 1.0))
        box[3] = round(min(box[1] + side - 1.0, height - 1.0))
    return results

def NMS(results, thresh, methodType):
    bboxes_nms = []
    if len(results) == 0:
        return bboxes_nms
    else:
        results = sorted(results, key=lambda result: result['bbox_score'], reverse=True)

    flag = np.zeros_like(results)
    for index, result_i in  enumerate(results):
        if flag[index] == 0:
            box_i = result_i['faceBox']
            area1 = (box_i[2] - box_i[0] + 1) * (box_i[3] - box_i[1] + 1)
            bboxes_nms.append(result_i)
            flag[index] = 1

            for j, result_j in enumerate(results):
                if flag[j] == 0:
                    box_j = result_j['faceBox']
                    area_intersect = (min(box_i[2], box_j[2]) - max(box_i[0], box_j[0]) + 1) * \
                             (min(box_i[3], box_j[3]) - max(box_i[1], box_j[1]) + 1)
                    if min(box_i[2], box_j[2]) - max(box_i[0], box_j[0]) < 0:
                        area_intersect = 0.0
                    area2 = (box_j[2] - box_j[0] + 1) * (box_j[3] - box_j[1] + 1)
                    iou = 0
                    if methodType == 'u':
                        iou = (area_intersect) * 1.0 / (area1 + area2 - area_intersect)
                    if methodType == 'm':
                        iou = (area_intersect) * 1.0 / min(area1, area2)
                    if iou > thresh:
                        flag[j] = 1
    return bboxes_nms



def GenerateBBox(confidence, reg, scale, threshold):
    ch, hs, ws = confidence.shape
    results = []
    for i in range(hs):
        for j in range(ws):
            if confidence[1][i][j] > threshold:
                result = {}
                box = []
                box.append(j * pnet_stride / scale)  # xmin
                box.append(i * pnet_stride / scale)  # ymin
                box.append((j * pnet_stride + pnet_cell_size - 1.0) / scale)  # xmax
                box.append((i * pnet_stride + pnet_cell_size - 1.0) / scale)  # ymax
                result['faceBox'] = box
                b_reg = []
                for k in range(reg.shape[0]):
                    b_reg.append(reg[k][i][j])
                result['bbox_reg'] = b_reg
                result['bbox_score'] = confidence[1][i][j]
                results.append(result)

    return results



def GetResult_net12(pnet, image ):
    image = (image.copy() - std_mean) * std_scale
    rows, cols, channels = image.shape
    scales = CalScale(cols, rows)

    results = []

    for scale in scales:
        ws = int(math.ceil(cols * scale))
        hs = int(math.ceil(rows * scale))
        scale_img = cv2.resize(image, (ws, hs), cv2.INTER_CUBIC)
        tempimg = np.zeros((1, hs, ws, 3))
        tempimg[0, :, :, :] = scale_img
        tempimg = tempimg.transpose(0, 3, 1, 2)

        pnet.blobs['data'].reshape(1, 3, hs, ws)
        pnet.blobs['data'].data[...] = tempimg

        pnet.forward()
        confidence = copy.deepcopy(pnet.blobs['prob1'].data[0])
        reg = copy.deepcopy(pnet.blobs['conv4-2'].data[0])

        result = GenerateBBox(confidence, reg, scale, pnet_thread)
        results.extend(result)

    res_boxes = NMS(results,0.7 ,'u')
    res_boxes = BBoxRegression(res_boxes)
    res_boxes = BBoxPadSquare(res_boxes, cols, rows)

    return res_boxes


def GetResult_net24(rnet, res_boxes, image):
    image = (image.copy() - std_mean) * std_scale
    lenth = len(res_boxes)
    num = int(math.floor(lenth * 1.0 / batchsize))
    rnet.blobs['data'].reshape(batchsize, 3, 24, 24)

    results = []

    if len(res_boxes) == 0:
        return results

    for i in range(num):
        tempimg = np.zeros((batchsize, 24, 24, 3))
        for j in range(batchsize):
            box = res_boxes[i * batchsize + j]['faceBox']
            box = np.array(box, dtype=np.int32)
            if box[2] > box[0] and box[3] > box[1]:
                roi = copy.deepcopy(image[box[1]: box[3], box[0]:box[2]])
                scale_img = cv2.resize(roi, (24, 24))
                tempimg[i, :, :, :] = scale_img

        tempimg = tempimg.transpose(0, 3, 1, 2)
        rnet.blobs['data'].data[...] = tempimg
        rnet.forward()
        confidence = copy.deepcopy(rnet.blobs['prob1'].data[...])
        reg = copy.deepcopy(rnet.blobs['conv5-2'].data[...])

        for j in range(batchsize):
            result = {}
            result['faceBox'] = res_boxes[i * batchsize + j]['faceBox']
            b_reg = []
            for k in range(reg.shape[1]):
                b_reg.append(reg[j][k])
            result['bbox_reg'] = b_reg
            result['bbox_score'] = confidence[j][1]
            if confidence[j][1] > onet_thread:
                results.append(result)


    resnum = lenth - num * batchsize
    if resnum > 0:
        rnet.blobs['data'].reshape(resnum, 3, 24, 24)
        tempimg = np.zeros((resnum, 24, 24, 3))
        for i in range(resnum):
            box = res_boxes[num * batchsize + i]['faceBox']
            box = np.array(box, dtype=np.int32)
            if box[2] > box[0] and box[3] > box[1]:
                roi = copy.deepcopy(image[box[1]: box[3], box[0]:box[2]])
                scale_img = cv2.resize(roi, (24, 24))
                tempimg[i, :, :, :] = scale_img

        tempimg = tempimg.transpose(0, 3, 1, 2)
        rnet.blobs['data'].data[...] = tempimg
        rnet.forward()
        confidence = copy.deepcopy(rnet.blobs['prob1'].data[...])
        reg = copy.deepcopy(rnet.blobs['conv5-2'].data[...])

        for i in range(resnum):
            result = {}
            result['faceBox'] = res_boxes[num * batchsize + i]['faceBox']
            b_reg = []
            for k in range(reg.shape[1]):
                b_reg.append(reg[i][k])
            result['bbox_reg'] = b_reg
            result['bbox_score'] = confidence[i][1]
            if confidence[i][1] > rnet_thread:
                results.append(result)

    res_boxes = NMS(results, 0.7, 'u')
    res_boxes = BBoxRegression(res_boxes)
    res_boxes = BBoxPadSquare(res_boxes, image.shape[1], image.shape[0])

    return res_boxes


def GetResult_net48(onet, res_boxes, image):
    image = (image.copy() - std_mean) * std_scale
    lenth = len(res_boxes)
    num = int(math.floor(lenth * 1.0 / batchsize))
    onet.blobs['data'].reshape(batchsize, 3, 48, 48)
    results = []

    if len(res_boxes) == 0:
        return results

    for i in range(num):
        tempimg = np.zeros((batchsize, 48, 48, 3))
        for j in range(batchsize):
            box = res_boxes[i * batchsize + j]['faceBox']
            box = np.array(box, dtype=np.int32)
            if box[2] > box[0] and box[3] > box[1]:
                roi = copy.deepcopy(image[box[1]: box[3], box[0]:box[2]])
                scale_img = cv2.resize(roi, (48, 48))
                tempimg[i, :, :, :] = scale_img

        tempimg = tempimg.transpose(0, 3, 1, 2)
        onet.blobs['data'].data[...] = tempimg
        onet.forward()
        confidence = copy.deepcopy(onet.blobs['prob1'].data[...])
        reg = copy.deepcopy(onet.blobs['conv6-2'].data[...])
        reg_landmark = copy.deepcopy(onet.blobs["conv6-3"].data[...])

        for j in range(batchsize):
            result = {}
            result['faceBox'] = res_boxes[i * batchsize + j]['faceBox']
            b_reg = []
            for k in range(reg.shape[1]):
                b_reg.append(reg[j][k])
            result['bbox_reg'] = b_reg
            result['bbox_score'] = confidence[j][1]

            w = result['faceBox'][2] - result['faceBox'][0] + 1
            h = result['faceBox'][3] - result['faceBox'][1] + 1
            l_reg = []
            for l in range(5):
                l_reg.append(reg_landmark[j][2 * l] * w + result['faceBox'][0])
                l_reg.append(reg_landmark[j][2 * l + 1] * h + result['faceBox'][1])

            result['landmark_reg'] = l_reg

            if confidence[j][1] > onet_thread:
                results.append(result)


    resnum = lenth - num * batchsize
    if resnum > 0:
        onet.blobs['data'].reshape(resnum, 3, 48, 48)
        tempimg = np.zeros((resnum, 48, 48, 3))
        for i in range(resnum):
            box = res_boxes[num * batchsize + i]['faceBox']
            box = np.array(box, dtype=np.int32)
            if box[2] > box[0] and box[3] > box[1]:
                roi = copy.deepcopy(image[box[1]: box[3], box[0]:box[2]])
                scale_img = cv2.resize(roi, (48, 48))
                tempimg[i, :, :, :] = scale_img

        tempimg = tempimg.transpose(0, 3, 1, 2)
        onet.blobs['data'].data[...] = tempimg
        onet.forward()
        confidence = copy.deepcopy(onet.blobs['prob1'].data[...])
        reg = copy.deepcopy(onet.blobs['conv6-2'].data[...])
        reg_landmark = copy.deepcopy(onet.blobs["conv6-3"].data[...])

        for i in range(resnum):
            result = {}
            result['faceBox'] = res_boxes[num * batchsize + i]['faceBox']
            b_reg = []
            for k in range(reg.shape[1]):
                b_reg.append(reg[i][k])
            result['bbox_reg'] = b_reg
            result['bbox_score'] = confidence[i][1]
            w = result['faceBox'][2] - result['faceBox'][0] + 1
            h = result['faceBox'][3] - result['faceBox'][1] + 1
            l_reg = []
            for k in range(int(reg_landmark.shape[1] / 2)):
                l_reg.append(reg_landmark[i][2 * k] * w + result['faceBox'][0])
                l_reg.append(reg_landmark[i][2 * k + 1] * h + result['faceBox'][1])

            result['landmark_reg'] = l_reg
            if confidence[i][1] > onet_thread:
                results.append(result)
    res_boxes = BBoxRegression(results)
    res_boxes = NMS(res_boxes, 0.7, 'm')
    res_boxes = BBoxPad(res_boxes, image.shape[1], image.shape[0])

    return res_boxes


def DetImage(pnet, rnet, onet, image, show = False):
    results = GetResult_net12(pnet, image)
    rnet_re = GetResult_net24(rnet, results, image)
    onet_re = GetResult_net48(onet, rnet_re, image)

    faceboxs = []
    for index, result in enumerate(onet_re):
        facebox = {}
        facebox['box'] = result['faceBox']
        facebox['landmark'] = result['landmark_reg']
        faceboxs.append(facebox)

        if show:
            cv2.rectangle(image, (int(facebox['box'][0]), int(facebox['box'][1])), (int(facebox['box'][2]), int(facebox['box'][3])),
                          (0, 0, 255), 1)
            for i in range(5):
                cv2.circle(image, (int(facebox['landmark'][2 * i]), int(facebox['landmark'][2 * i + 1])), 2, (55, 255, 155), -1)
    if show:
        cv2.imshow('', image)
        cv2.waitKey(0)

    return faceboxs


if __name__ == "__main__":

    root = './model/'
    caffe.set_device(0)
    caffe.set_mode_gpu()
    pnet = caffe.Net(root + 'det1.prototxt', root + 'det1.caffemodel', caffe.TEST)
    rnet = caffe.Net(root + 'det2.prototxt', root + 'det2.caffemodel', caffe.TEST)
    onet = caffe.Net(root + 'det3.prototxt', root + 'det3.caffemodel', caffe.TEST)

    image = cv2.imread('./timg02.jpeg')

    results = DetImage(pnet, rnet, onet, image, False)

print 'results: ', results
