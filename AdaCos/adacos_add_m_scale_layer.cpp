#include <algorithm>
#include <vector>
#include <math.h>
#include "caffe/layers/adacos_add_m_scale_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void AdaCosAddmScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const AdaCosAddmScaleParameter& param = this->layer_param_.adacos_add_m_scale_param();
    m_ = param.m();
    num_classes = param.num_classes();
    sin_m = sin(m_);
    cos_m = cos(m_);
    threshold = cos(M_PI - m_);
    s_d = sqrt(2) * log(num_classes - 1);
    cos_theta_med = cos(M_PI / 4);
    transform_test_ = param.transform_test() & (this->phase_ == TRAIN);
  }

  template <typename Dtype>
  void AdaCosAddmScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
    top_flag.ReshapeLike(*bottom[0]);
    cos_theta.ReshapeLike(*bottom[0]);
    Bi_.ReshapeLike(*bottom[0]);
  }

template <typename Dtype>
void AdaCosAddmScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* tpflag = top_flag.mutable_cpu_data();
  Dtype* cos_t = cos_theta.mutable_cpu_data();
  Dtype* bi_data = Bi_.mutable_cpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, top_data);
  caffe_copy(count, bottom_data, cos_t);
  caffe_set(count, Dtype(0), tpflag);
  caffe_set(count, Dtype(1.0), bi_data); 

  //cosin add m
  for (int i = 0; i < num; ++i) {
    int gt = static_cast<int>(label_data[i]);
    if(gt < 0) continue;
    
    bi_data[i * dim + gt] = 0.f;

    Dtype cos_theta_2 = cos_t[i * dim + gt] * cos_t[i * dim + gt];
    Dtype sin_theta = sqrt(1.0f - cos_theta_2);
    if(cos_t[i * dim + gt] > 1.0f)
    {
        LOG(INFO) << "cos_theta > 1 ****** " << cos_t[i * dim + gt];
        cos_t[i * dim + gt] = 1.0f;
        cos_theta_2 = 1.0f;
        sin_theta = 0.0f;
    }

    if(cos_t[i * dim + gt] <= threshold)
    {
        top_data[i * dim + gt] = cos_t[i * dim + gt] - sin(M_PI - m_) * m_;
        tpflag[i * dim + gt] = 1.0f;
    }
    else
        top_data[i * dim + gt] = cos_t[i * dim + gt] * cos_m - sin_theta * sin_m;

//    if(count_num % 10 == 0)
//    {
//        LOG(INFO) << "top_data[" << i * dim + gt << "]: " << top_data[i * dim + gt] << "  cos_t: " << cos_t[i * dim + gt];
//    }
  }
  
  
  //compute cos_theta_med
  Dtype sum_theta = 0.f;
  for(int i = 0; i < count; i++)
  {
      float arccos_x = acos(bottom_data[i]);
      sum_theta += arccos_x;
  }
  sum_theta = sum_theta / count;
  sum_theta = std::min(double(sum_theta), M_PI / 4);
  cos_theta_med = cos(sum_theta);

  //compute s_d
  caffe_mul(count, bottom_data, bi_data, bi_data);
  caffe_scal(count, s_d, bi_data);
  caffe_exp(count, bi_data, bi_data);
  s_d = log(caffe_cpu_asum(count, bi_data) / num) / cos_theta_med;

  //scale
  caffe_scal(count, s_d, top_data);
}

template <typename Dtype>
void AdaCosAddmScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
    caffe_scal(count, s_d, bottom_diff);

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

      if(tpflag[i * dim + gt] > 0.0f)
        coffe = 1.0f;
      bottom_diff[i * dim + gt] = coffe * bottom_diff[i * dim + gt];

//      if(count_num_back % 10 == 0)
//      {
//          LOG(INFO) << "top_diff: " << top_diff[i * dim + gt];
//          LOG(INFO) << "bottom_diff: " << bottom_diff[i * dim + gt];
//          LOG(INFO) << "cos_theta[ "<<i * dim + gt <<"]: "<< cos_t[i * dim + gt]<<  "    coffe: " << coffe;
//      }
    }
  }
}

INSTANTIATE_CLASS(AdaCosAddmScaleLayer);
REGISTER_LAYER_CLASS(AdaCosAddmScale);

}  // namespace caffe

