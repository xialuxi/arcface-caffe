#include <algorithm>
#include <vector>
#include <math.h>
#include "caffe/layers/sv-x-softmax.hpp"

namespace caffe {

  template <typename Dtype>
  void CosinAddmLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const CosinAddmParameter& param = this->layer_param_.cosin_add_m_param();
    m_ = param.m();
    t_ = param.t();
    sin_m = sin(m_);
    cos_m = cos(m_);
    threshold = cos(M_PI - m_);
    transform_test_ = param.transform_test() & (this->phase_ == TRAIN);
  }

  template <typename Dtype>
  void CosinAddmLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
    top_flag.ReshapeLike(*bottom[0]);
    cos_theta.ReshapeLike(*bottom[0]);
  }

template <typename Dtype>
void CosinAddmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* tpflag = top_flag.mutable_cpu_data();
  Dtype* cos_t = cos_theta.mutable_cpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, top_data);
  caffe_copy(count, bottom_data, cos_t);
  caffe_set(count, Dtype(0), tpflag);

  for (int i = 0; i < num; ++i) {
    int gt = static_cast<int>(label_data[i]);
    if(gt < 0) continue;
    Dtype cos_theta_2 = cos_t[i * dim + gt] * cos_t[i * dim + gt];
    Dtype sin_theta = sqrt(1.0f - cos_theta_2);
    if(cos_t[i * dim + gt] > 1.0f)
    {
        LOG(INFO) << "cos_theta > 1 ****** " << cos_t[i * dim + gt];
        cos_t[i * dim + gt] = 1.0f;
        cos_theta_2 = 1.0f;
        sin_theta = 0.0f;
    }

    if(cos_t[i * dim + gt] <= threshold) {
        top_data[i * dim + gt] = cos_t[i * dim + gt] - sin(M_PI - m_) * m_;
        tpflag[i * dim + gt] = 1.0f;
    }
    else {
        top_data[i * dim + gt] = cos_t[i * dim + gt] * cos_m - sin_theta * sin_m;
    }

    for (j = 0; j < dim; j++)
    {
      if(j == gt) continue;
      if ( top_data[i * dim + j] > top_data[i * dim + gt] ) {
        top_data[i * dim + j] = top_data[i * dim + j] * t_ + t_ - 1;
        tpflag[i * dim + j] = 2.0f;
      }
    }

  }
}

template <typename Dtype>
void CosinAddmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* label_data = bottom[1]->cpu_data();
    const Dtype* cos_t = cos_theta.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* tpflag = top_flag.cpu_data();
    int count = bottom[0]->count();

    caffe_copy(count, top_diff, bottom_diff);

    int num = bottom[0]->num();
    int dim = count / num;
    for (int i = 0; i < num; ++i)
    {
      int gt = static_cast<int>(label_data[i]);
      if(gt < 0) continue;
      Dtype cos_theta_2 = cos_t[i * dim + gt] * cos_t[i * dim + gt];
      if(cos_t[i * dim + gt] == 1.0f)
      {
          cos_theta_2 = 1.0f;
      }
      Dtype sin_theta = sqrt(1.0f - cos_theta_2);
      Dtype coffe = 0.0f;
      if(sin_theta == 0.0f)
          coffe = 1.0f;
      else
        coffe = cos_m + sin_m * cos_t[i * dim + gt] / sin_theta;

      if(tpflag[i * dim + gt] == 1.0f)
        coffe = 1.0f;
      bottom_diff[i * dim + gt] = coffe * top_diff[i * dim + gt];
      
      for (j = 0; j < dim; j++)
      {
        if(j == gt) continue;
        if ( tpflag[i * dim + j] == 2.0f) {
          bottom_diff[i * dim + j] = bottom_diff[i * dim + j] * t_;
        }
      }
    
    }
  }
}

INSTANTIATE_CLASS(CosinAddmLayer);
REGISTER_LAYER_CLASS(CosinAddm);

}  // namespace caffe

