# (Deep) Machine Learning 

A collection of (deep) machine learning papers, tutorial, datasets and projects.

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


## Video tutorials

- [Analytics+Vidhya: Data Science tutorials](https://www.analyticsvidhya.com/blog/2016/12/30-top-videos-tutorials-courses-on-machine-learning-artificial-intelligence-from-2016/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29)


## Learning algorithms

[The Marginal Value of Adaptive Gradient Methods in Machine Learning](https://arxiv.org/pdf/1705.08292v1.pdf):
Adaptive Gradient Methods such as Adam converge faster and might even achieve better training error but
have worse test error than Stochastic Gradient Decent.


## Augmentation

[mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412): 
Linear interpolation with random factor between samples in a batch. Interpolation is on input AND output data, which requires
that data is numerical, e.g. images and one-hot-encoded labels and that loss function can handle no-binary labels.
Easy to implement, paper shows good results. [Reviewer comments](https://openreview.net/forum?id=r1Ddp1-Rb) are interesting as well.
Supported in [nuts-ml](https://maet3608.github.io/nuts-ml/nutsml.html#nutsml.batcher.Mixup)


## Regularization

[Concrete Dropout](https://arxiv.org/pdf/1705.07832v1.pdf):
Automatic tuning of the dropout probability using gradient methods.


## Segmentation

[Dense Transformer Networks](https://arxiv.org/pdf/1705.08881v1.pdf):
Automatic learning of patch sizes and shapes in contrast to fixed, rectangular pixel centered patches
for segmentation. Achieves better segmentation.


## Saliency maps

[Real Time Image Saliency for Black Box Classifiers](https://arxiv.org/pdf/1705.07857v1.pdf):
Fast saliency detection method that can be applied to any differentiable image classifier.


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

[DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055): Search for
network architectures using gradient decent. Considerably faster and simpler than other methods.

[Evolving simple programs for playing Atari games](https://arxiv.org/abs/1806.05695): Search for
network architectures/image processing functions using Cartesian Genetic Programming.


## Understanding Networks / Visualization

[Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf): Shows how
utilize a Global Average Pooling layer to compute so called Class Activation Maps (CAM) that allow to identify
regions of the input image that are important for the classification result.

[Grad-CAM: Why did you say that?](https://arxiv.org/abs/1611.07450): An extension of the above paper
that enables the computation of Class Activation Maps (CAM) for arbitrary entwork architectures.

[t-SNE-CUDA: GPU-Accelerated t-SNE and its Applications to Modern Data](https://arxiv.org/abs/1807.11824):
A CUDA implementation of t-sne that is substantially faster than other implementation and for instance
allows to visualize the entire ImageNet data set.


