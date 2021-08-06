# Siamese Adversarial Network for Image Classification of Heavy Mineral Grains

Huizhen Hao<sup>a, b</sup>, Zhiwei Jiang<sup>c</sup>, Shiping Gec<sup>c</sup>, Cong Wangc<sup>c</sup>, Qing Gu<sup>c</sup>

<sup>a</sup> Software Institute, Nanjing University, Nanjing 210023, China

<sup>b</sup> School of Information and Communication Engineering, Nanjing Institute of Technology, Nanjing 211167, China

 <sup>c </sup>State Key Laboratory for Novel Software Technology, Nanjing University, Nanjing 210023, China



This repository contains the source code and dataset to perform prediction and evaluation. 

## Content

### Data

​	The image datasets of the Yangtze River , Yarlung Zangbu River  and PumQu River 

### Traditional_features

+ the traditional features abstracted from the  Yangtze River cross-polarized training set

### util

- config.py

  define dict-like container that allows for attribute-based access to keys.

- DomainImageFolder.py

  the public method

- util.py

  the public method


### Model

+ train_resnet18.py

  train baseline model resnet18 on  the training set

+ test_resnet18.py

  evaluate baseline model resnet18 on  the test set 

+ train_resnet34.py

  train   baseline model resnet34 on  the training set

+ test_resnet34.py

  evaluate baseline model resnet34  on the testset

+ train_vgg16.py

  train  baseline model vgg16 on  the training set

+ test_vgg16.py

  evaluate baseline model vgg16 on  the test set

+ SAN.py

  Siamese Adversarial Network definition

+ train_SAN.py

  adversarial train SAN  on  the training set

+ test_SAN.py

  evaluate SAN on the test set 



## Running the files

train_SAN.py --config config/stage1.yaml  #traine on ChangJiang YaJiang 

test_SAN.py --config config/stage1.yaml #test on ChangJiang YaJiang 



## License

The following legal note is restricted solely to the content of the named files. It cannot
overrule licenses from the Python standard distribution modules, which are imported and
used therein.

BSD 3-clause license

Copyright (c) 2021 Huizhen Hao, Zhiwei Jiang, Shiping Ge, Cong Wang, Qing Gu.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the names of the copyright holders nor the names of any contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.
