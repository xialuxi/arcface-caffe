#include <algorithm>
#include <vector>
#include <cmath>
#include "caffe/layers/cosin_add_m_layer.hpp"


namespace caffe {

  template <typename Dtype>
  __global__ void CosinAddmForward(const int n, const int dim, const Dtype* label,
                                                 Dtype* top_data, Dtype threshold, Dtype bais, Dtype* flag) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
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
  __global__ void CosinAddmBackward(const int n, const int dim, const Dtype* label,
                                                 Dtype* bottom_diff, const Dtype* cos_data, Dtype bais, const Dtype* flag) {
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      if(flag[index * dim + gt] == 0.0f) {
            Dtype cos_theta = cos_data[index * dim + gt];
            Dtype sin_theta = sqrt(1 - pow(cos_theta,2));
            bottom_diff[index * dim + gt] = bottom_diff[index * dim + gt] *(cos(bais) + sin(bais) * cos_theta / sin_theta);
        }
    }
  }

  template <typename Dtype>
  void CosinAddmLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* tpflag = top_flag.mutable_gpu_data();
    Dtype* cos_t = cos_theta.mutable_gpu_data();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);
    if (!transform_test_ && this->phase_ == TEST) return;

    caffe_copy(count, bottom_data, top_data);
    caffe_copy(count, bottom_data, cos_t);
    caffe_gpu_set(count, Dtype(0), tpflag);

    // NOLINT_NEXT_LINE(whitespace/operators)
    CosinAddmForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, label_data, top_data, threshold, m_, tpflag);
    CUDA_POST_KERNEL_CHECK;

    const Dtype* cos_test = cos_theta.cpu_data();
    const Dtype* tpflag_test = top_flag.cpu_data();
    const Dtype* top_data_test = top[0]->cpu_data();
  }

  template <typename Dtype>
  void CosinAddmLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                     const vector<bool>& propagate_down,
                                                     const vector<Blob<Dtype>*>& bottom) {
    if (top[0] != bottom[0] && propagate_down[0]) {

      int num = bottom[0]->num();
      int count = bottom[0]->count();
      int dim = count / num;

      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* label_data = bottom[1]->gpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

      caffe_copy(count, top_diff, bottom_diff);

      const Dtype* tpflag = top_flag.gpu_data();
      const Dtype* cos_t = cos_theta.gpu_data();

      CosinAddmBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
        num, dim, label_data, bottom_diff, cos_t, m_, tpflag);
      CUDA_POST_KERNEL_CHECK;

    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(CosinAddmLayer);
}  // namespace caffe
