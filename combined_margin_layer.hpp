#ifndef CAFFE_COMBINED_MARGIN_LAYER_HPP_
#define CAFFE_COMBINED_MARGIN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class CombinedMarginLayer : public Layer<Dtype> {
 public:
  explicit CombinedMarginLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CombinedMargin"; }
  virtual inline int MinNumBottomBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype m1;
  Dtype m2;
  Dtype m3;
  Blob<Dtype> m1_arccos_x_add_m2;
  bool transform_test_;
};

}  // namespace caffe

#endif  // CAFFE_COMBINED_MARGIN_LAYER_HPP_

