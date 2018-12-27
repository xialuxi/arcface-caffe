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

9、增加mxnet中带SE结构的网络模型转化为caffemodel的方法


