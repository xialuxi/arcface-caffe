# arcface-caffe

1、在caffe中实现arcface中的损失函数，参考cosinface的实现。

2、caffe工程，以及其它层的实现请参考：https://github.com/xialuxi/AMSoftmax

3、编译过程：

    (1)下载https://github.com/xialuxi/AMSoftmax工程，修改好make.config的环境配置
    
    (2)将cosin_add_m_layer.hpp拷贝到目录： ./caffe/include/caffe/layers/下
    
    (3)将cosin_add_m_layer.cpp 、 cosin_add_m_layer.cu 拷贝到目录： ./caffe/src/caffe/layers/下
    
    (4)根据proto文件，对应修改./caffe/src/caffe/proto/caffe.proto文件
    
    (5)make -j

4、原理请参考：https://github.com/deepinsight/insightface

5、增加Combined Margin Loss 参考insightface的实现

6、增加mtcnn人脸检测python代码，根据c++代码改写，效果没有任何损失，模型与原始代码请参考：https://github.com/blankWorld/MTCNN-Accelerate-Onet

7、实际训练的时候，caffe的收敛速度慢而且困难，而mxnet的速度则比较快，具体原因还不清楚，解决方法参考：https://github.com/xialuxi/arcface-caffe/issues/7

8、增加 insightface的gpu实现代码

9、增加mxnet中带SE结构的网络模型转化为caffemodel的方法，几乎无精度损失。

10、增加人脸关键点检测损失函数wing_loss代码， 以及人脸关键点和姿态估计的网络和预训练模型。
    论文：https://arxiv.org/abs/1711.06753v4
    
11、增加基于梯度均衡的损失函数，可以替换softmax，传送门：https://github.com/xialuxi/GHMLoss-caffe

12、更正cosin_add_m_layer.cu反向传播的计算，谢谢 @zhaokai5 的指正。

13、 更新了新的关键点检测和人脸姿态估计模型, 模型大小不到1M.

14、 增加SV-X-Softmax的实现.参考论文:  《Support Vector Guided Softmax Loss for Face Recognition》

15、 增加AdaCos的实现， 参考论文《AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations》

16， 新的人脸检测算法（包含关键点检测）：RetinaFace
     链接：https://github.com/xialuxi/insightface/tree/master/RetinaFace


17, 基于centernet的人脸检测算法以及关键点检测挺不错的，实现也很简单， 看下效果图(resnet18)：
只需要添加关键点回归的分支即可，参考：https://github.com/xingyizhou/CenterNet

![Image text](https://github.com/xialuxi/arcface-caffe/blob/master/face_detection/0.jpg)!





















18 、　本人最近在做k8s的kubeflow的分布式部署和训练平台，暂停有关人脸的更新，带来不便，敬请谅解！
