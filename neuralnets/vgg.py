"""
Convolutional neural networks are an important class of learnable representations applicable, 
among others, to numerous computer vision problems. Deep CNNs, in particular, are composed of 
several layers of processing, each involving linear as well as non-linear operators, that are 
learned jointly, in an end-to-end manner, to solve a particular tasks. These methods are now 
the dominant approach for feature extraction from audiovisual and textual data.

This practical explores the basics of learning (deep) CNNs. The first part introduces typical 
CNN building blocks, such as ReLU units and linear filters, with a particular emphasis on 
understanding back-propagation. The second part looks at learning two basic CNNs. The first 
one is a simple non-linear filter capturing particular image structures, while the second one 
is a network that recognises typewritten characters (using a variety of different fonts). These 
examples illustrate the use of stochastic gradient descent with momentum, the definition of an 
objective function, the construction of mini-batches of data, and data jittering. The last part 
shows how powerful CNN models can be downloaded off-the-shelf and used directly in applications, 
bypassing the expensive training process.
"""
