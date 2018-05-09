Multilayer Perceptron(MLP)
===
Tensorflow implementaion of MLP for classification of MNIST_dataset. Basic code for studying.

Requirement
---
1. Tensorflow 1.4.0
2. Python 3.5.4
3. Python packages : numpy, matplotlib, os, argparse

Usage
---
### Command
`python main.py`

### Arguments
Optional
* `--layer1` : Number of layer1 nodes in MLP. Default : `64`
* `--layer2` : Number of layer2 nodes in MLP. Default : `128`
* `--layer3` : Number of layer3 nodes in MLP. Default : `256`
* `--epoch` : Number of epochs to run. Default : `10`
* `--batch_size` : Number of batch_size to run. Default : `50`
* `--learning_rate` : Learning rate for Adam optimizer. Default : `0.001`
* `--drop_rate` : Prob of dropout. Default : `0.7`
* `--disp_num` : How many display MNIST prediction. Default : `5`

Results
---
`python main.py`

training

![result1](/image/result_train.PNG)

Prediction number and test image

![result2](/image/result_fig.PNG)

Reference Implementations
---
+ https://github.com/golbin/TensorFlow-ML-Exercises




