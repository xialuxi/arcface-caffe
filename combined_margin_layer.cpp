#include <algorithm>
#include <vector>
#include <math.h>
#include "caffe/layers/combined_margin_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void CombinedMarginLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) 
{
    const CombinedMarginParameter& param = this->layer_param_.combined_margin_param();
    m1 = param.m1();
    m2 = param.m2();
    m3 = param.m3();
    transform_test_ = param.transform_test() & (this->phase_ == TRAIN);
  }

  template <typename Dtype>
  void CombinedMarginLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
    m1_arccos_x_add_m2.ReshapeLike(*bottom[0]);
  }

template <typename Dtype>
void CombinedMarginLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* m1_x_m2 = m1_arccos_x_add_m2.mutable_cpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, top_data);
  //acos(x)
  for(int i = 0; i < count; i++)
  {
      float arccos_x = acos(bottom_data[i]);
      m1_x_m2[i] = m1 * arccos_x + m2;
  }


  for (int i = 0; i < num; ++i)
  {
    int gt = static_cast<int>(label_data[i]);
    if(gt < 0) continue;

    top_data[i * dim + gt] = cos(m1_x_m2[i * dim + gt]) -m3;
  }
}

template <typename Dtype>
void CombinedMarginLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom)
{
  if (propagate_down[0])
  {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* label_data = bottom[1]->cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* m1_x_m2 = m1_arccos_x_add_m2.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int count = bottom[0]->count();

    caffe_copy(count, top_diff, bottom_diff);

    int num = bottom[0]->num();
    int dim = count / num;
    for (int i = 0; i < num; ++i)
    {
      int gt = static_cast<int>(label_data[i]);
      if(gt < 0) continue;
      Dtype diff_gt = m1 * pow(1 - pow(bottom_data[i * dim + gt], 2), -0.5) * sin(m1_x_m2[i * dim + gt]);
      bottom_diff[i * dim + gt] *= diff_gt;
    }
  }
}

INSTANTIATE_CLASS(CombinedMarginLayer);
REGISTER_LAYER_CLASS(CombinedMargin);

}  // namespace caffe
