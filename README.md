<!--
 Copyright 2022 Victor I. Afolabi

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
# TensorFlow Playground

> This repository is developed for educational purposes using the official
TensorFlow [tutorials] & [guides].

[tutorials]: https://www.tensorflow.org/tutorials
[guides]: https://www.tensorflow.org/guide

This consists of various Machine Learning algorithms and AI concepts developed
with TensorFlow.

Each task (or tutorial) has it's own Notebook or project files both as a Notebook
and as a Python module.

## Beginner

The "Hello, World!" for Machine Learning to introduce the Keras Sequential API
and `model.fit`.

- [ ] **ML basics with Keras**

  - [x] Basic image classification
  - [ ] Basic text classification
  - [ ] Text classification with TF Hub
  - [ ] Regression
  - [ ] Overfit and underfit
  - [ ] Save and Load
  - [ ] Tune hyperparameters with the Keras Tuner

- [ ] **Load and preprocess data**

  - [ ] Images
  - [ ] Videos
  - [ ] CSV
  - [ ] NumPy
  - [ ] pandas.DataFrame
  - [ ] TFRecord and tf.Example
  - [ ] Text

## Advanced

This introduces more advanced TensorFlow concepts including model subclassing,
custom training loops, custom layers, distributed training accross muultiple GPUs,
multiple machines or TPUs.

- [ ] **Customization**

  - [ ] Tensors and operations
  - [ ] Custom layers
  - [ ] Custom training: walkthrough

- [ ] **Distributed training**
  - [ ] Distributed training with Keras
  - [ ] Custom training loops
  - [ ] Multi-worker training with Keras
  - [ ] Multi-worker training with CTL
  - [ ] Save and load
  - [ ] Distributed input

- [ ] **Vision**
  - [ ] Convolutional Neural Network
  - [ ] Image classification
  - [ ] Trainsfer learning and fine-tunning
  - [ ] Trainsfer learning with TF Hub
  - [ ] Data Augmentation
  - [ ] Image segmentation
  - [ ] Video classification
  - [ ] Transfer lerarning with MoViNet

- [ ] **Text**
  - [ ] Wordembeddings
  - [ ] Word2Vec
  - [ ] Warm start embedding matrix with changing vocabulary
  - [ ] Classify text with BERT
  - [ ] Translate text with Transformer
  - [ ] Image captioning

- [ ] **Audio**
  - [ ] Simple audio recognition
  - [ ] Transfer learning for audio recognition
  - [ ] Generae music with an RNN

- [ ] **Structured data**
  - [ ] Classify structured data with preprocessing layers
  - [ ] Classify on imbalanced data
  - [ ] Time series forcasting
  - [ ] Decision forest models
  - [ ] Recommenders

- [ ] **Generative**
  - [ ] Stable Diffusion
  - [ ] Neural Style Transfer
  - [ ] CycleGAN
  - [ ] Adversarial FGSM
  - [ ] Intro to Autoencoder
  - [ ] Variational Autoencoder
  - [ ] Lossy data compression

- [ ] **Model optimization**
  - [ ] Scale mmodel compression with EPR
  - [ ] TensorFlow model optimization

- [ ] **Model Understanding**
  - [ ] Integrated gradients
  - [ ] Uncertainty quantification with SNGP
  - [ ] Probabilistic regression

- [ ] **Reinforcement learning**
  - [ ] Actor-Critic method
  - [ ] TensorFlow agents

## Contribution

You are very welcome to modify and use them in your own projects.

Please keep a link to the [original repository]. If you have made a fork with
substantial modifications that you feel may be useful, then please [open a new
issue on GitHub][issues] with a link and short description.

## License (Apache)

This project is opened under the [Apache License 2.0][license] which allows very
broad use for both private and commercial purposes.

A few of the images used for demonstration purposes may be under copyright.
These images are included under the "fair usage" laws.

[original repository]: https://github.com/victor-iyi/tf-playground
[issues]: https://github.com/victor-iyi/tf-playground/issues
[license]: ./LICENSE
