1、人脸关键点检测，以及人脸姿态估计。


2、网络参考： https://github.com/tensor-yu/cascaded_mobilenet-v2


3、训练使用的损失函数为： （1）关键点使用wing loss 。 （2）方向使用L2loss


4、训练使用的数据为： umdfaces_batch3、 umdfaces_batch2, （需要翻墙）下载连接：https://www.umdfaces.io/


5、提供的预训练模型，满足大多数情况（大角度下，姿态比关键点准）。
 
 方向依次为，（1）pitch(上正下负) （2）yaw(左正右负) （3）roll(左正右负)


6、caffe.proto文件中添加：

   optional WingLossParameter wing_loss_param = 158;

   message WingLossParameter {
  optional float w = 1 [default = 10];
  optional float epsilon = 2 [default = 2];
}
