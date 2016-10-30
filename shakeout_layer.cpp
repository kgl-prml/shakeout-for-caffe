// Created by Guoliang Kang. 

#include <vector>
#include <iostream>

#include "caffe/layers/shakeout_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer_factory.hpp"
#include <math.h>
#include <algorithm>

namespace caffe {

template <typename Dtype>
void ShakeoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.shakeout_param().dropout_ratio();
  DCHECK(threshold_ >= 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);

  float shakeout_scale = this->layer_param_.shakeout_param().shakeout_scale();
  //DCHECK(shakeout_scale >= 0.);
  //DCHECK(shakeout_scale < 1.);
  shakeout_const_ = shakeout_scale; //shakeout_scale / (2.0*(1-shakeout_scale));
  //rescale_param_ = 1 + this->threshold_ / (2*(1-this->threshold_));
  //LOG(INFO) << "rescale param: " << this->rescale_param_;
  tanh_smooth_scale_ = this->layer_param_.shakeout_param().tanh_smooth_scale();
  operate_layer_type_ = this->layer_param_.shakeout_param().operate_layer_type();
  CHECK(operate_layer_type_ != "Shakeout")
      << "The operating layer type should not be 'Shakeout'"; 

  // initialize the operating virtual layer
  LayerParameter operate_layer_param(this->layer_param_);
  operate_layer_param.set_type(this->operate_layer_type_);
  operate_layer_param.clear_shakeout_param();  
  operate_layer_ = LayerRegistry<Dtype>::CreateLayer(operate_layer_param);
  operate_layer_->SetUp(bottom, top);
  // initialize the Shakeout learnable parameters and its sign vector
  this->blobs_ = operate_layer_->blobs();
  // reshape sign_blobs_
  this->sign_blobs_.resize(this->blobs_.size());
  // Currently we do not apply Shakeout on the bias terms
  for(int i = 0; i < this->sign_blobs_.size(); i++) {
     this->sign_blobs_[i].reset(new Blob<Dtype>((this->blobs_[i])->shape()));
     caffe_set(this->sign_blobs_[i]->count(), Dtype(0), this->sign_blobs_[i]->mutable_cpu_data());
     caffe_set(this->sign_blobs_[i]->count(), Dtype(0), this->sign_blobs_[i]->mutable_cpu_diff());
  }
}

template <typename Dtype>
void ShakeoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.reset(new Blob<Dtype>()); 

  rand_vec_->Reshape(bottom[0]->shape());
  // reshape operating layer
  operate_layer_->Reshape(bottom, top);
}

template <typename Dtype>
void ShakeoutLayer<Dtype>::generateMask_cpu(){
    Dtype* mask = rand_vec_->mutable_cpu_data();
    caffe_rng_uniform<Dtype>(rand_vec_->count(), 0, 1, mask);
    int count = rand_vec_->count();
    for(int i=0;i<count;i++){
	 mask[i] = mask[i]>=threshold_?1.:0.;
    }
    //Dtype* mask_dependent = rand_vec_[1]->mutable_cpu_data();
    //for(int i=0;i<count;i++){
    //   mask_dependent[i] = shakeout_const_ * (mask[i] - 1.0);
    //}
}

template <typename Dtype>
void ShakeoutLayer<Dtype>::getSignOfWeight_cpu(){
    const Dtype* weights = this->blobs_[0]->cpu_data();
    Dtype* sign_weights = this->sign_blobs_[0]->mutable_cpu_data();
    const int count = (this->blobs_)[0]->count();
    for(int i=0;i<count;i++){
       sign_weights[i] = weights[i]>0?1:(weights[i]<0?-1:0);
    }
    if(this->sign_blobs_.size()>1){
      caffe_set(this->sign_blobs_[1]->count(), Dtype(0), this->sign_blobs_[1]->mutable_cpu_data());
    }
}

template <typename Dtype>
void ShakeoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count_bottom = bottom[0]->count();
  const int count_top = top[0]->count();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    generateMask_cpu();
    // the signs of biases are all set zeros
    getSignOfWeight_cpu();

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    vector<Blob<Dtype>*> bottom_split1(bottom.size()), bottom_split2(bottom.size());
    for(int i = 0; i < bottom.size(); i++){
        bottom_split1[i] = new Blob<Dtype>(bottom[i]->shape());
        bottom_split2[i] = new Blob<Dtype>(bottom[i]->shape());
    }
    Dtype* bottom_split1_data = bottom_split1[0]->mutable_cpu_data();
    Dtype* bottom_split2_data = bottom_split2[0]->mutable_cpu_data();

    vector<Blob<Dtype>*> top_split1(top.size()), top_split2(top.size());
    for(int i = 0; i < top.size(); i++){
        top_split1[i] = new Blob<Dtype>(top[i]->shape());
        top_split2[i] = new Blob<Dtype>(top[i]->shape());
    }

    const Dtype* mask = rand_vec_->mutable_cpu_data();
    for (int i = 0; i < count_bottom; ++i) {
      bottom_split1_data[i] = bottom_data[i] * mask[i] * scale_;
    }

    // reset the operate_layer_'s weights
    operate_layer_->blobs() = this->blobs_;
    // forward  
    operate_layer_->Forward(bottom_split1, top_split1);
    // reset and forward again
    for (int i = 0; i < count_bottom; ++i) {
      Dtype tmp_scale = shakeout_const_ * (mask[i] * scale_ - 1);
      bottom_split2_data[i] = bottom_data[i] * tmp_scale;
    }
    operate_layer_->blobs() = this->sign_blobs_; 
    operate_layer_->Forward(bottom_split2, top_split2);
    // sum
    const Dtype* top_split1_data = top_split1[0]->cpu_data();
    const Dtype* top_split2_data = top_split2[0]->cpu_data();
    for (int i = 0; i < count_top; ++i) {
      top_data[i] = top_split1_data[i] + top_split2_data[i];
    }
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
void ShakeoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
 
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();

    const Dtype* mask = rand_vec_->cpu_data();

    const int count_bottom = bottom[0]->count();

    if (this->phase_ == TRAIN) {
       vector<Blob<Dtype>*> bottom_split1(bottom.size()), bottom_split2(bottom.size());
       for(int i = 0; i < bottom.size(); i++){
          bottom_split1[i] = new Blob<Dtype>(bottom[i]->shape());
          bottom_split2[i] = new Blob<Dtype>(bottom[i]->shape());
       }
       Dtype* bottom_split1_data = bottom_split1[0]->mutable_cpu_data();
       const Dtype* bottom_split1_diff = bottom_split1[0]->cpu_diff();
       Dtype* bottom_split2_data = bottom_split2[0]->mutable_cpu_data();
       const Dtype* bottom_split2_diff = bottom_split2[0]->cpu_diff();
       for (int i = 0; i < count_bottom; ++i) {
         bottom_split1_data[i] = bottom_data[i] * mask[i] * scale_;
       }
       for (int i = 0; i < count_bottom; ++i) {
         bottom_split2_data[i] = bottom_data[i] * shakeout_const_
		 * (mask[i] * scale_ - 1);
       }
 
       // compute gradients with respect to neurons
       // reset the operate_layer_'s weights
       operate_layer_->blobs() = this->blobs_;
       // backward  

       operate_layer_->Backward(top, propagate_down, bottom_split1);
       // reset and backward again
       operate_layer_->blobs() = this->sign_blobs_; 
       for(int i=0;i<this->sign_blobs_.size();i++){
	   caffe_set(this->sign_blobs_[i]->count(), Dtype(0), this->sign_blobs_[i]->mutable_cpu_diff());
       }
       operate_layer_->Backward(top, propagate_down, bottom_split2);

       for (int i = 0; i < count_bottom; ++i) {
         bottom_diff[i] = bottom_split1_diff[i] * mask[i] * scale_
        		+ bottom_split2_diff[i] * shakeout_const_ * (mask[i] * scale_ - 1);
       }

       // compute gradients with respect to weights and biases (for biases, already updated in the previous process)
       Dtype* weights_diff = this->blobs_[0]->mutable_cpu_diff();
       const Dtype* weights_data = this->blobs_[0]->cpu_data();
       Dtype* sign_weights_diff = this->sign_blobs_[0]->mutable_cpu_diff();

       for (int i=0; i< this->blobs_[0]->count(); i++){
	   Dtype sign_diff_approx = tanh_smooth_scale_ * (1 - 
		pow(tanh(tanh_smooth_scale_ * weights_data[i]), 2.0));
           sign_weights_diff[i] *= sign_diff_approx;
           weights_diff[i] += sign_weights_diff[i];
       }
       for(int i = 0; i < bottom.size(); i++){
         delete bottom_split1[i];
         delete bottom_split2[i];
       }
    } else {
      //caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
}


#ifdef CPU_ONLY
STUB_GPU(ShakeoutLayer);
#endif

INSTANTIATE_CLASS(ShakeoutLayer);
REGISTER_LAYER_CLASS(Shakeout);

}  // namespace caffe
