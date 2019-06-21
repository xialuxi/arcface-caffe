#include "caffe/layers/wing_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void WingLossForward(const int n, const Dtype* abs_d, const Dtype* log_d, 
    const float w, const float epsilon, const float c) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype abs_val = abs_d[index];
    if (abs_val < w) {
      log_d[index] = w * log_d[index];
    } else {
      log_d[index] = abs_val - c;
    }
  }
}

template <typename Dtype>
void WingLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    
    Dtype* sub_x_data = diff_.mutable_gpu_data();
    Dtype* abs_x_data = abs_x.mutable_gpu_data();
    Dtype* log_abs_data = log_abs.mutable_gpu_data();

    int count = bottom[0]->count();

    caffe_set(count, Dtype(0), sub_x_data);
    caffe_set(count, Dtype(0), abs_x_data);
    caffe_set(count, Dtype(1.0), log_abs_data);

    caffe_sub(count, bottom_data, label_data, sub_x_data);
    caffe_abs(count, sub_x_data, abs_x_data);
    const Dtype scale = Dtype(1.0 / epsilon);
    caffe_axpy(count, scale, abs_x_data, log_abs_data);
    caffe_log(count, log_abs_data, log_abs_data);
    caffe_scal(count, w, log_abs_data);

    Dtype loss = 0.f;
    WingLossForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, abs_x.mutable_gpu_data(), log_abs.mutable_gpu_data(),
      w, epsilon, _c);
    CUDA_POST_KERNEL_CHECK;
    
    caffe_gpu_dot(count, log_abs.gpu_data(), one_dot.gpu_data(), &loss);
    top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
}

template <typename Dtype>
__global__ void WingLossBackward(const int n, const Dtype* in, Dtype* out,
    const float w, const float epsilon) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    Dtype sign = (Dtype(0) < val) - (val < Dtype(0));
    if (abs_val < w) {
      out[index] = sign * w / (epsilon + abs_val) ;
    } else {
      out[index] = sign;
    }
  }
}

template <typename Dtype>
void WingLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    const Dtype* sub_x_data = diff_.gpu_data();
    const Dtype* abs_x_data = abs_x.gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  
    int count = bottom[0]->count();

  WingLossBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, sub_x_data, bottom_diff,
      w, epsilon);
  CUDA_POST_KERNEL_CHECK;

  if (propagate_down[0]) {
    Dtype loss_weight = top[0]->gpu_diff()[0];
    caffe_scal(count, loss_weight / bottom[0]->num(), bottom_diff);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(WingLossLayer);

} // namespace caffe
