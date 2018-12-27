
1、普通网络转换参考代码：  https://github.com/GarrickLin/MXNet2Caffe


2、SE网络结构在人脸识别中表现比较优秀，提供带SE的网络模型转化代码，以及示例。


3、caffe新添加层：

    （1）、 将axpy_layer.hpp拷贝到目录： ./caffe/include/caffe/layers/下

    (2)、 将axpy_layer.cpp 、 axpy_layer.cu 拷贝到目录： ./caffe/src/caffe/layers/下
    
    (3)、 make -j

4、具体过程：

    （1）、 json2prototxt.py代码，将mxnet的模型描述文件model-symbol.json，转化为caffe的网络描述文件face.prototxt.
    
    (2)、 修改转化不正确的地方，例如：
         对生成的prototxt文件搜索bottom: “_mulscalar0”，将_mulscalar0改为上一层的”data”，也就是将bottom: “_mulscalar0”改为bottom: “data”
         
    （3）、 将elemwise_add操作，使用caffe的 Axpy层代替，具体怎么修改，请参考face.prototxt（SErenet50）
    
    （4）、 利用 http://ethereon.github.io/netscope/#/editor 检查网络是否连接正确
    
    （5）、 使用代码mxnet_caffe.py转换模型。
    
    （6）、 精度测试

5、将bn层改成inplace的写法，节约显存。

6、合并bn层参数到卷积层中，加速计算。
