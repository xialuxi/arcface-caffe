#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/wing_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void WingLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    w = this->layer_param_.wing_loss_param().w();
    DCHECK(w > 0);
    epsilon = this->layer_param_.wing_loss_param().epsilon();
    DCHECK(epsilon >= 0);


    //_c = w * (1.0 - log(1.0 + w/epsilon)
    _c = w * (1.0 - log(1.0 + w/epsilon));

   // has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
   // if (has_ignore_label_)
   // {
   //     ignore_label_ = this->layer_param_.loss_param().ignore_label();
   // }
    log_abs.ReshapeLike(*bottom[0]);
    caffe_set(bottom[0]->count(), Dtype(1.0), one_dot_data);
}

template <typename Dtype>
void WingLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    LossLayer<Dtype>::Reshape(bottom, top);

    //x = landmarks - labels
    diff_.ReshapeLike(*bottom[0]);
    abs_x.ReshapeLike(*bottom[0]);
    log_abs.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void WingLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
   
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label_data = bottom[1]->cpu_data();
    
    Dtype* sub_x_data = diff_.mutable_cpu_data();
    Dtype* abs_x_data = abs_x.mutable_cpu_data();
    Dtype* log_abs_data = log_abs.mutable_cpu_data();

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
    for(int i = 0; i < count; i++)
    {
        if(w > abs_x_data[i])
        {
            loss += log_abs_data[i];
        }
        else
        {
            loss += abs_x_data[i] - _c;
        }
    }
    
    loss = loss / bottom[0]->num();
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void WingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom)
{
    const Dtype* sub_x_data = diff_.cpu_data();
    const Dtype* abs_x_data = abs_x.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
    int count = bottom[0]->count();

    if (propagate_down[0])
    {
         for(int i = 0; i < count; i++)
        {
            Dtype sign = (Dtype(0) < sub_x_data[i]) - (sub_x_data[i] < Dtype(0));

            if(w > abs_x_data[i])
            {
                bottom_diff[i] = sign * w / (abs_x_data[i] + epsilon);
            }
            else
            {
                bottom_diff[i] = sign; 
            }
        }
    }

    Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / bottom[0]->num(), bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(WingLossLayer);
#endif

INSTANTIATE_CLASS(WingLossLayer);
REGISTER_LAYER_CLASS(WingLoss);

}  // namespace caffe
