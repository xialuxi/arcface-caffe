#ifndef CAFFE_ADACOS_ADD_M_SCALE_LAYER_HPP_
#define CAFFE_ADACOS_ADD_M_SCALE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace caffe {

template <typename Dtype>
class AdaCosAddmScaleLayer : public Layer<Dtype> {
 public:
  explicit AdaCosAddmScaleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AdaCosAddmScale"; }
  virtual inline int MinNumBottomBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype m_;
  Dtype sin_m;
  Dtype cos_m;
  Dtype threshold;
  Blob<Dtype> cos_theta;
  Blob<Dtype> top_flag;
  bool transform_test_;

  Dtype num_classes;
  Blob<Dtype> Bi_;
  Dtype cos_theta_med;
  Dtype s_d;
};

}  // namespace caffe

#endif  // CAFFE_ADACOS_ADD_M_SCALE_LAYER_HPP_

