# shakeout-for-caffe

This Shakeout implementation is based on **caffe**.

First, you should download the **caffe** source code.

Then put the files **shakeout_layer.cpp**; **shakeout_layer.cu** under the directory **src/caffe/layers/**;
put the file **caffe.proto** under the directory **src/caffe/proto/**; 
put the file **shakeout_layer.hpp** under the directory **include/caffe/layers/**.

Finally, you can refer to **AlexNet_shakeout_example.prototxt** to see how to use.                                                       

Please cite our Shakeout paper in your publications if it helps your research:

```
@inproceedings{kang2016shakeout,
  title={Shakeout: A New Regularized Deep Neural Network Training Scheme},
  author={Kang, Guoliang and Li, Jun and Tao, Dacheng},
  booktitle={Thirtieth AAAI Conference on Artificial Intelligence},
  year={2016}
}
```
