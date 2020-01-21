# (Deep) Machine Learning 

A collection of (deep) machine learning papers, tutorial, datasets and projects.

## Other paper/link collections

[NeuroIPS](https://github.com/hindupuravinash/nips2017)
[The incredible Pytorch](https://www.ritchieng.com/the-incredible-pytorch/)


## Projects

[Unsupervised mapping of bird sounds](https://experiments.withgoogle.com/ai/bird-sounds): T-SNE applied
to a large set of bird sounds to visualize the similarity/dissimilarity of bird songs. Beautiful 
visualization that is great fun to play with.

[Quickdraw](https://quickdraw.withgoogle.com): Recognition of quickly drawn sketches of things.


## Datasets

[Quickdraw](https://github.com/googlecreativelab/quickdraw-dataset): A collection of 50 million drawings across 
345 categories, contributed by players of the game Quick, Draw. Also some fascinating analytics of the
dataset.


## Talks

- [Bay Area ML talks Day 1](https://www.youtube.com/watch?v=eyovmAtoUx0)
- [Bay Area ML talks Day 2](https://www.youtube.com/watch?v=9dXiAecyJrY)


## Overview / Critical reviews

- [Deep Learning: A Critical Appraisal](https://arxiv.org/abs/1801.00631v1): A very nice summary of the issues with the current
state of Deep Learning.


## Video tutorials

- [Analytics+Vidhya: Data Science tutorials](https://www.analyticsvidhya.com/blog/2016/12/30-top-videos-tutorials-courses-on-machine-learning-artificial-intelligence-from-2016/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29)

- [Introduction to ML by Facebook](https://research.fb.com/the-facebook-field-guide-to-machine-learning-video-series/)


## Tutorials

[Machine Learning: Basic Principles](https://arxiv.org/abs/1805.05052v4)

[Deep Learning - Berkely](https://berkeley-deep-learning.github.io/cs294-131-s17/)

[Tensorflow](https://github.com/chiphuyen/stanford-tensorflow-tutorials/)

[More Tensorflow](https://github.com/nlintz/TensorFlow-Tutorials)

[Reinforcement Learning1](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
[Reinforcement Learning2](https://github.com/dennybritz/reinforcement-learning)
[Reinforcement Learning3](http://www.argmin.net/2018/06/25/outsider-rl/)


## Blogs

[Rules of Machine Learning: Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml/)


## Learning improvements

[The Marginal Value of Adaptive Gradient Methods in Machine Learning](https://arxiv.org/pdf/1705.08292v1.pdf):
Adaptive Gradient Methods such as Adam converge faster and might even achieve better training error but
have worse test error than Stochastic Gradient Decent.

[Faster Convergence & Generalization in DNNs](https://arxiv.org/abs/1807.11414v2): An SVM based training on mini batches
that reduces the number of epochs by several magnitudes. Effect on training time is unclear since SVM training is performed on CPU
though there are GPU-based implementations. Learned networks are shown to be more robust to adversarial noise and over-fitting. 

[All You Need is Beyond a Good Init: Exploring Better Solution for Training Extremely Deep Convolutional Neural Networks with Orthonormality and Modulation](https://arxiv.org/abs/1703.01827): The authors propose a regularizer variant that allows to
train very deep networks without the need for residual (shortcuts/identity mappings) connections.

[Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks](https://arxiv.org/abs/1806.05393): The authors demonstrate that it is possible to train vanilla CNNs with ten thousand layers 
or more by using an appropriate initialization scheme. Implementation is available for tensorflow and [PyTorch](https://github.com/tanhongweibest/CNN).

[Understanding Batch Normalization](https://arxiv.org/abs/1806.02375v2): An analysis of batch normalization that reveals that
batch normalization has a regularizing effect that improves generalization of normalized networks. Activations become large and the convolutional channels become increasingly ill-behaved for layers deep in unnormalized networks.

[Recent Advances in Convolutional Neural Network Acceleration](https://arxiv.org/abs/1807.08596v1): A nice review of methods to accelerate CNN training and inference (including hardware approacjes).

[Reducing Parameter Space for Neural Network Training](https://arxiv.org/pdf/1805.08340v2): Authors show that limiting the network weights to be on a hypersphere leads to better result for some small regression problems and is less sensitive to the initial
weight initialization.

[Do Deep Nets Really Need to be Deep?](https://arxiv.org/pdf/1312.6184.pdf): A very interesting paper demonstrating that shallow
networks can perform as well as deep networks - rasing the question, whether DL architectures are actually necessary.


## Augmentation

[mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412): 
Linear interpolation with random factor between samples in a batch. Interpolation is on input AND output data, which requires
that data is numerical, e.g. images and one-hot-encoded labels and that loss function can handle no-binary labels.
Easy to implement, paper shows good results. [Reviewer comments](https://openreview.net/forum?id=r1Ddp1-Rb) are interesting as well.
Supported in [nuts-ml](https://maet3608.github.io/nuts-ml/nutsml.html#nutsml.batcher.Mixup)

[Manifold Mixup: Encouraging Meaningful On-Manifold Interpolation as a Regularizer](https://arxiv.org/abs/1806.05236):
An improved version of mixup where mixup performed not only in input space but also on the network internal layer outputs.


## Regularization

[Regularization and Optimization strategies in Deep Convolutional Neural Network](https://arxiv.org/pdf/1712.04711.pdf): 
A nice summary of regularization techniques and deep learning in general.

[Concrete Dropout](https://arxiv.org/pdf/1705.07832v1.pdf):
Automatic tuning of the dropout probability using gradient methods.


## Segmentation

[Dense Transformer Networks](https://arxiv.org/pdf/1705.08881v1.pdf):
Automatic learning of patch sizes and shapes in contrast to fixed, rectangular pixel centered patches
for segmentation. Achieves better segmentation.


## Unsupervised

[Look, Listen and Learn](https://arxiv.org/pdf/1705.08168v1.pdf):
Learning from unlabelled video and audio data.


## Semi-Supervised

[Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf):
Very good results for semi-supervised training on MNIST, CIFAR-10 and SVHN datasets.


## Variational Autoencoders

[Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks](https://arxiv.org/abs/1701.04722):
New training procedure for Variational Autoencoders based on adversarial training.


## GANs

[Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf):
Very good results for semi-supervised training on MNIST, CIFAR-10 and SVHN datasets.

[Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks](https://arxiv.org/abs/1701.04722):
New training procedure for Variational Autoencoders based on adversarial training.


## Reinforcement Learning

[Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/pdf/1705.08439v1.pdf):
Decomposes the problem into separate planning and generalisation tasks and shows better performance than 
Policy Gradients.


## Architecture search

[Neural Architecture Search: A Survey](https://arxiv.org/pdf/1808.05377v1.pdf): As the title says:
a review of methods to automatically determine the structure of a neural network.

[DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055): Search for
network architectures using gradient decent. Considerably faster and simpler than other methods.

[Evolving simple programs for playing Atari games](https://arxiv.org/abs/1806.05695): Search for
network architectures/image processing functions using Cartesian Genetic Programming.


## Understanding Networks / Visualization

[Real Time Image Saliency for Black Box Classifiers](https://arxiv.org/pdf/1705.07857v1.pdf):
Fast saliency detection method that can be applied to any differentiable image classifier.

[Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150): Shows how
utilize a Global Average Pooling layer to compute so called Class Activation Maps (CAM) that allow to identify
regions of the input image that are important for the classification result.

[Grad-CAM: Why did you say that?](https://arxiv.org/abs/1611.07450): An extension of the above paper
that enables the computation of Class Activation Maps (CAM) for arbitrary entwork architectures.

[RISE: Randomized Input Sampling for Explanation of Black-box Models](https://arxiv.org/abs/1806.07421):
A black-box approach that uses randomly occluded images to create saliency maps. More accurate than
Grad-CAM, does not require a specific network architecture and creates high-resolution maps.

[t-SNE-CUDA: GPU-Accelerated t-SNE and its Applications to Modern Data](https://arxiv.org/abs/1807.11824):
A CUDA implementation of t-sne that is substantially faster than other implementation and for instance
allows to visualize the entire ImageNet data set.

[Identifying Weights and Architectures of Unknown ReLU Networks](https://arxiv.org/abs/1910.00744):
An extremely cool paper showing that it is possible to reconstruct the architecture, weights, and biases 
of a deep ReLU network given the ability to query the network. 

## Tensor Factorization

[A general model for robust tensor factorization with unknown noise](https://arxiv.org/abs/1705.06755): Impressive
de-nosing of images using tensor factorization.

[Canonical Tensor Decomposition for Knowledge Base Completion](https://arxiv.org/abs/1806.07297): 
A nice comparison of tensor-based methods for Knowledge Base Completion demonstrating that CP
performs as well as others provided parameters are chosen carefully.


## Neuro-symbolic computing

Comination/integration of symbolic reasoning/representation with sub-symbolic/distributed representations.

[Neural-Symbolic Learning and Reasoning: A Survey and Interpretation](https://arxiv.org/abs/1711.03902): 
How to integrate low-level, sub-symbolic neural network learning and high-level, symbolic reasoning.


### Visual Question Answering

[Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding](https://arxiv.org/abs/1810.02338):
A nice introduction and reference of VQA approaches. Introducing a novel method (easy to understand but less "organic") 
with excellent results (99.8%) on CLEVR.

[FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871): Highly
accurate (on CLEVR) and simple model (FiLM).

[How clever is the FiLM model, and how clever can it be?](https://arxiv.org/abs/1809.03044): An analysis
of the FiLM model for VQA.

[A simple neural network module for relational reasoning](https://arxiv.org/abs/1706.01427): A beautifully
simple network architecture for relational learning and VQ answering but less accurate than FiLM.

[Inferring and executing programs for visual reasoning](https://arxiv.org/abs/1705.03633): Uses LSTMs to creates 
programs from questions to perform symbolic reasoning on scene (CLEVR benchmark).

[CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](https://arxiv.org/abs/1612.06890):
Introducing the CLEVR benchmark. A piece of art!

[An Analysis of Visual Question Answering Algorithms](https://arxiv.org/abs/1703.09684): A useful overview
over VQ datasets (missing CLEVR, however) 

[Explainable Neural Computation via Stack Neural Module Networks](https://arxiv.org/abs/1807.08556v1): 
Neural module networks for visual question answering.


### Relational Learning

[A Review of Relational Machine Learning for Knowledge Graphs](https://arxiv.org/abs/1503.00759): A very nice
review of Relational Learning using tensor factorization and neural network approaches (e.g. RESCAL).

[A Three-Way Model for Collective Learning on Multi-Relational Data](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwiy2qmQpZbfAhUN3Y8KHbufAIgQFjAAegQIChAC&url=http%3A%2F%2Fwww.icml-2011.org%2Fpapers%2F438_icmlpaper.pdf&usg=AOvVaw0frK-gzkLllcW0uapkn4Lp): Tensor-based Relational Learning
introducing the simple but effective RESCAL algorithm.

[Holographic Embeddings of Knowledge Graphs](https://arxiv.org/abs/1510.04935): An improvement of RESCAL that
leads to a model with considerably less parameters and better accuracy.

[Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/abs/1412.6575): A
very nice comparison of embedding based relational learning algorithms.

[A simple neural network module for relational reasoning](https://arxiv.org/abs/1706.01427): A beautifully
simple network architecture for relational learning and VQ answering.

[FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871): A slighly older more
complex 

[A Semantic Matching Energy Function for Learning with Multi-relational Data](https://arxiv.org/abs/1301.3485):
Relational Network learns relationship embeddings and combines entity embeddings and relationship embeddings 
separately first before combining the results.

[A latent factor model for highly multi-relational data](https://papers.nips.cc/paper/4744-a-latent-factor-model-for-highly-multi-relational-data.pdf):
An improved method for relational learning with performance better than RESCAL and SME, 
which scales well for large numbers of relationships.

