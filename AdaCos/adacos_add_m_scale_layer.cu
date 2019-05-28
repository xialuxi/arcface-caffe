#include <algorithm>
#include <vector>
#include <math.h>
#include "caffe/layers/adacos_add_m_scale_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void AdaCosinAddmForward(const int n, const int dim, const Dtype* label,
                                                 Dtype* top_data, Dtype threshold, Dtype bais, Dtype* flag, Dtype* bi_data) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      bi_data[index * dim + gt] = 0.f;
      if(top_data[index * dim + gt] < 1.0f) {
          Dtype theta = acos(top_data[index * dim + gt]);
          if (top_data[index * dim + gt] > threshold) {
            top_data[index * dim + gt] = cos(theta + bais);
        }
        else
        {
            top_data[index * dim + gt] = top_data[index * dim + gt] - bais * sin(bais);
            flag[index * dim + gt] = 1.0f;
        }
      }
    }
  }

  template <typename Dtype>
  __global__ void AdaCosinAddmBackward(const int n, const int dim, const Dtype* label,
                                                 Dtype* bottom_diff, const Dtype* cos_data, Dtype bais, const Dtype* flag) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      if(flag[index * dim + gt] == 0.0f) {
            Dtype cos_theta = cos_data[index * dim + gt];
            Dtype sin_theta = sqrt(1 - pow(cos_theta,2));
            bottom_diff[index * dim + gt] *= (cos(bais) + sin(bais) * cos_theta / sin_theta);
        }
    }
  }

template <typename Dtype>
  __global__ void ComputeAcos(const int n, Dtype* input_data) {   
    CUDA_KERNEL_LOOP(index, n) {
        input_data[index] = (Dtype)acos(input_data[index]);
    }
  }

template <typename Dtype>
void AdaCosAddmScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* label_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* tpflag = top_flag.mutable_gpu_data();
  Dtype* cos_t = cos_theta.mutable_gpu_data();
  Dtype* bi_data = Bi_.mutable_gpu_data();

  Dtype* mutable_bottom_data = bottom[0]->mutable_gpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, top_data);
  caffe_copy(count, bottom_data, cos_t);
  caffe_gpu_set(count, Dtype(0), tpflag);
  caffe_gpu_set(count, Dtype(1.0), bi_data); 

  AdaCosinAddmForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, label_data, top_data, threshold, m_, tpflag, bi_data);
  CUDA_POST_KERNEL_CHECK;
  

  //compute cos_theta_med
  ComputeAcos<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (num, mutable_bottom_data);
  CUDA_POST_KERNEL_CHECK;
  
  Dtype avg_theta;
  caffe_gpu_asum(count, mutable_bottom_data, &avg_theta);
  avg_theta = avg_theta / count;
  avg_theta = std::min(double(avg_theta), M_PI / 4);
  cos_theta_med = cos(avg_theta);

  //compute s_d
  caffe_gpu_mul(count, cos_t, bi_data, bi_data);
  caffe_gpu_scal(count, s_d, bi_data);
  caffe_gpu_exp(count, bi_data, bi_data);
  caffe_gpu_asum(count, bi_data, &s_d);
  s_d = log(s_d / num) / cos_theta_med;

  //recovery bottom_data for debugging and visualization
  caffe_copy(count, cos_t, mutable_bottom_data);

  //scale
  caffe_gpu_scal(count, s_d, top_data);
}

template <typename Dtype>
void AdaCosAddmScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* label_data = bottom[1]->gpu_data();
    const Dtype* cos_t = cos_theta.gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* tpflag = top_flag.gpu_data();

    int count = bottom[0]->count();
    int num = bottom[0]->num();
    int dim = count / num;

    caffe_copy(count, top_diff, bottom_diff);
    caffe_gpu_scal(count, s_d, bottom_diff);

    AdaCosinAddmBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, label_data, bottom_diff, cos_t, m_, tpflag);
      CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AdaCosAddmScaleLayer);

}  // namespace caffe

