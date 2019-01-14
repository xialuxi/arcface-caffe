1、人脸关键点检测，以及人脸姿态估计。


2、网络参考： https://github.com/tensor-yu/cascaded_mobilenet-v2


3、训练使用的损失函数为： （1）关键点使用wing loss 。 （2）方向使用L2loss


4、训练使用的数据为： umdfaces_batch3、 umdfaces_batch2


5、提供的预训练模型，满足大多数情况（大角度下，姿态比关键点准）。
