#coding=utf-8
import sys, argparse
import numpy as np
import mxnet as mx
import cv2
sys.path.insert(0,'./caffe-master/python')
import caffe
import copy


def get_model(image_size, model_path, epoch, layer):
    ctx = mx.gpu(0)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model



def get_feature(model, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # color transform:BGR---RGB
    img = np.transpose(img, (2, 0, 1))
    input_blob = np.expand_dims(img, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()[0]
    l2_norm = cv2.norm(embedding, cv2.NORM_L2)
    return embedding / l2_norm

def get_caffe_net(caffemodel, prototxt):
    facenet = caffe.Net(prototxt, caffemodel, caffe.TEST)
    return facenet

def caffe_get_feature(net, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img - 127.5
    img = img * 0.0078125
    tempimg = np.zeros((1, 112, 112, 3))
    tempimg[0, :, :, :] = img
    tempimg = tempimg.transpose(0, 3, 1, 2)
    net.blobs['data'].data[...] = tempimg
    net.forward()
    features = copy.deepcopy(net.blobs['fc1'].data[...])
    feature = np.array(features[0])
    l2_norm = cv2.norm(feature, cv2.NORM_L2)
    return feature / l2_norm


def cos(v1, v2):
    lenth = len(v1)
    sum = 0
    for i in range(lenth):
        sum += v1[i] * v2[i]
    return sum


if __name__ == '__main__':


    # mxnet init
    model_path = './r50_model/model'
    epoch = 11
    image_size = (112,112)
    layer = 'fc1'
    model = get_model(image_size, model_path, epoch, layer)



    #caffe net init
    caffemodel = './r50_model/face.caffemodel'
    prototxt = './r50_model/face.prototxt'
    net = get_caffe_net(caffemodel, prototxt)


    image_path = './0_0.jpg'

    img = cv2.imread(image_path)
    feature_mxnet = get_feature(model, img)
    feature_caffe = caffe_get_feature(net, img)

    print 'sim: ', cos(feature_mxnet, feature_caffe)

