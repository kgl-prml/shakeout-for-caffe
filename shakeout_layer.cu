// Created by Guoliang Kang.

#include <vector>
#include <algorithm>
#include "caffe/layers/shakeout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ShakeoutForward_kernel(const int n,
    const Dtype scale, const Dtype shakeout_const,
    const Dtype* mask, const Dtype* in,
    Dtype* out1, Dtype* out2) {
  CUDA_KERNEL_LOOP(index, n) {
    out1[index] = in[index] * mask[index] * scale;
    out2[index] = in[index] * shakeout_const *
	(mask[index] * scale - 1);
  }
}

template <typename Dtype>
__global__ void generateMask_kernel(const int count, Dtype* mask, const Dtype threshold){
  CUDA_KERNEL_LOOP(index, count) {
     mask[index] = mask[index]>=threshold?1:0;
  }
}

template <typename Dtype>
void ShakeoutLayer<Dtype>::generateMask_gpu(){
    Dtype* mask = rand_vec_->mutable_gpu_data();
    caffe_gpu_rng_uniform<Dtype>(rand_vec_->count(), 0, 1,mask);
    int count = rand_vec_->count();
    generateMask_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, mask, this->threshold_);
}

template <typename Dtype>
__global__ void getSignOfWeight_kernel(const int count, const Dtype* data, Dtype* sign_data, const Dtype scale){
  CUDA_KERNEL_LOOP(index, count) {
     sign_data[index] = data[index]>0?1:(data[index]<0?-1:0);
  }
}

template <typename Dtype>
void ShakeoutLayer<Dtype>::getSignOfWeight_gpu(){
    const Dtype* weights = this->blobs_[0]->gpu_data();
    Dtype* sign_weights = this->sign_blobs_[0]->mutable_gpu_data();
    const int count = this->blobs_[0]->count();
    getSignOfWeight_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, weights, sign_weights, this->tanh_smooth_scale_);
    if(this->sign_blobs_.size()>1){
      caffe_gpu_set(this->sign_blobs_[1]->count(), Dtype(0), this->sign_blobs_[1]->mutable_gpu_data());
    }
}

template <typename Dtype>
__global__ void TruncWeight_kernel(Dtype* data, const int count, const float threshold){
      CUDA_KERNEL_LOOP(index, count) {
        if(data[index] < threshold && data[index] > -threshold){
            data[index] = 0;
        }   
      }
}

template <typename Dtype>
void ShakeoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count_bottom = bottom[0]->count();
  const int count_top = top[0]->count();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    generateMask_gpu();
    // the signs of biases are all set zeros
    getSignOfWeight_gpu();

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    vector<Blob<Dtype>*> bottom_split1(bottom.size()), bottom_split2(bottom.size());
    for(int i = 0; i < bottom.size(); i++){
        bottom_split1[i] = new Blob<Dtype>(bottom[i]->shape());
        bottom_split2[i] = new Blob<Dtype>(bottom[i]->shape());
    }
    Dtype* bottom_split1_data = bottom_split1[0]->mutable_gpu_data();
    Dtype* bottom_split2_data = bottom_split2[0]->mutable_gpu_data();

    vector<Blob<Dtype>*> top_split1(top.size()), top_split2(top.size());
    for(int i = 0; i < top.size(); i++){
        top_split1[i] = new Blob<Dtype>(top[i]->shape());
        top_split2[i] = new Blob<Dtype>(top[i]->shape());
    }
    Dtype* top_split1_data = top_split1[0]->mutable_gpu_data();
    Dtype* top_split2_data = top_split2[0]->mutable_gpu_data();

    const Dtype* mask = rand_vec_->mutable_gpu_data();

    ShakeoutForward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(count_bottom, this->scale_, this->shakeout_const_, mask, bottom_data, bottom_split1_data, bottom_split2_data);
    // reset the operate_layer_'s weights
    operate_layer_->blobs() = this->blobs_;
    // forward  
    operate_layer_->Forward(bottom_split1, top_split1);
    // reset and forward again
    operate_layer_->blobs() = this->sign_blobs_; 
    operate_layer_->Forward(bottom_split2, top_split2);
    // sum
    // gpu version
    caffe_gpu_add<Dtype>(count_top, top_split1_data, top_split2_data, top_data);
   
    for(int i = 0; i < bottom.size(); i++){
      delete bottom_split1[i];
      delete bottom_split2[i];
    }
    for(int i = 0; i < top.size(); i++){
      delete top_split1[i];
      delete top_split2[i];
    }

  } else {
    operate_layer_->blobs() = this->blobs_;
    operate_layer_->Forward(bottom, top);
  }
}

template <typename Dtype>
__global__ void ShakeoutBackward_neuron_kernel(const int n,
    const Dtype scale, const Dtype shakeout_const,
    const Dtype* in_diff1, const Dtype* in_diff2,
    const Dtype* mask, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff1[index] * mask[index] * scale
	 + in_diff2[index] * shakeout_const *
	 (mask[index] * scale - 1);
  }
}

template <typename Dtype>
__global__ void ShakeoutBackward_weights_kernel(const int n,
    Dtype* sign_weights_diff, const Dtype* sign_weights_data,
    Dtype* weights_diff, const Dtype scale, const Dtype* weights_data){
  CUDA_KERNEL_LOOP(index, n) {
     Dtype sign_diff_approx = scale * (1 - 
	tanh(scale * weights_data[index]) * tanh(scale * weights_data[index]));
     sign_weights_diff[index] *= sign_diff_approx;
     weights_diff[index] += sign_weights_diff[index];
  }
}

template <typename Dtype>
void ShakeoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
 
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();

    const Dtype* mask = rand_vec_->gpu_data();

    const int count_bottom = bottom[0]->count();
    if (this->phase_ == TRAIN) {
       vector<Blob<Dtype>*> bottom_split1(bottom.size()), bottom_split2(bottom.size());
       for(int i = 0; i < bottom.size(); i++){
          bottom_split1[i] = new Blob<Dtype>(bottom[i]->shape());
          bottom_split2[i] = new Blob<Dtype>(bottom[i]->shape());
       }
       Dtype* bottom_split1_data = bottom_split1[0]->mutable_gpu_data();
       const Dtype* bottom_split1_diff = bottom_split1[0]->gpu_diff();
       Dtype* bottom_split2_data = bottom_split2[0]->mutable_gpu_data();
       const Dtype* bottom_split2_diff = bottom_split2[0]->gpu_diff();
 
    ShakeoutForward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(count_bottom, this->scale_, this->shakeout_const_, mask, bottom_data, bottom_split1_data, bottom_split2_data);

       // compute gradients with respect to neurons
       // reset the operate_layer_'s weights
       operate_layer_->blobs() = this->blobs_;

       // backward  
       operate_layer_->Backward(top, propagate_down, bottom_split1);
       // reset and backward again
       operate_layer_->blobs() = this->sign_blobs_; 
       for(int i=0;i<this->sign_blobs_.size();i++){
	   caffe_gpu_set(this->sign_blobs_[i]->count(), Dtype(0), this->sign_blobs_[i]->mutable_gpu_diff());
       }

       operate_layer_->Backward(top, propagate_down, bottom_split2);

       // gpu version
       ShakeoutBackward_neuron_kernel<Dtype><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(count_bottom, this->scale_, this->shakeout_const_, bottom_split1_diff, bottom_split2_diff, mask, bottom_diff);

       // compute gradients with respect to weights and biases (for biases, already updated in the previous process
       Dtype* weights_diff = this->blobs_[0]->mutable_gpu_diff();
       const Dtype* weights_data = this->blobs_[0]->gpu_data();
       Dtype* sign_weights_diff = this->sign_blobs_[0]->mutable_gpu_diff();
       const Dtype* sign_weights_data = this->sign_blobs_[0]->gpu_data();
       ShakeoutBackward_weights_kernel<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(this->blobs_[0]->count(), sign_weights_diff, sign_weights_data, weights_diff, this->tanh_smooth_scale_, weights_data);
       for(int i = 0; i < bottom.size(); i++){
         delete bottom_split1[i];
         delete bottom_split2[i];
       }
    } else {
      //caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(ShakeoutLayer);


}  // namespace caffe
