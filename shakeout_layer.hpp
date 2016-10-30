#ifndef CAFFE_SHAKEOUT_LAYERS_HPP_
#define CAFFE_SHAKEOUT_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/** Shakeout layer
 *
 */
template <typename Dtype>
class ShakeoutLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides ShakeoutParameter shakeout_param,
   *     with ShakeoutLayer options:
   *   - dropout_ratio (\b optional, default 0.5);
   *   - shakeout_scale (\b optional, default 0.5);
   *   - tanh_smooth_scale (\b optional, default 1).
   */
  explicit ShakeoutLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Shakeout"; }
  inline const std::string operate_layer_type() const { return this->operate_layer_type_; }
  inline const vector< shared_ptr<Blob<Dtype> > >& sign_blobs(){ return sign_blobs_; }
  inline const shared_ptr<Blob<Dtype> >& rand_vec() { return rand_vec_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void generateMask_cpu(); 
  void generateMask_gpu(); 
  void getSignOfWeight_cpu();
  void getSignOfWeight_gpu();
  // Shakeout related settings
  shared_ptr< Blob<Dtype> > rand_vec_;
  Dtype threshold_;
  Dtype scale_;
  Dtype shakeout_const_;
  Dtype tanh_smooth_scale_;
  Dtype rescale_param_;
  Dtype trunc_ratio_;
  // operating layer type
  std::string operate_layer_type_;
  shared_ptr< Layer<Dtype> > operate_layer_;
  vector<shared_ptr<Blob<Dtype> > > sign_blobs_;
}; //end shakeout layer definition


}  // namespace caffe

#endif  // CAFFE_SHAKEOUT_LAYERS_HPP_
