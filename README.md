# Draco
Open source low-level deep learning framework for C++ and Julia.


This is a purely pointer-arithemetic implementation of the framework.
 Weights and biases are stored in as 2D and 1D arrays on the heap. The neural network is an object of the Model class (similar to the Model class
 in both Keras and Pytorch), which implements a doubly linked list of dense hidden layers. 
 
 The forward pass of the nextwork performs linear transformations on each successive output vector, and calculates the loss function for that 
 epoch as a function of the output vector of the final layer of the network. Backpropagation is achieved by traversing the list in the opposite 
 direction, starting with the output layer and ending with updating the first hidden layer. The gradient of the loss with respect to each layer
 is kept track of, and is updated by taking an elementwise product with the vector derivative of the activation function for each layer; the 
 gradient in weight of a given layer is then given by the tensor product of the gradient (chain_deriv in code) with the input vector to that layer.
 The gradient in bias is simply equal to the updated chain derivative itself, since d(Wx + b)/db = 1 and so you're only multiplying the chain derivative
 by 1.
 Backpropagating to the previous layer requires applying the chain rule; the partial derivative of the linear transformation of a given dense layer
 of form y = Wx + b wrt the input vector to the layer x is simply the transpose of W; then the updated chain rule going into the previous layer
 is going to be W_transpose * chain_deriv. The process in the previous layer repeats the same, all the way to the first layer of the network.
 
 
 # Future of Draco 
 Additional changes will be made over time. Currently (as of today),Draco only supports dense hidden layers.
 I will seek to add dropout layers and convolutions (hopefully with the help of other programmers since this was tricky already!)
 In addition, currently this version is constrained to regression models and binary classifiers. I hope to also introduce a vector output layer 
 with one-hot-encoding and a softmax activation to allow for multilabel classification.
 
 I don't expect this project to "blow up" by any means. There are many libraries that already handle machine learning model deployment 
 well in C++ (Caffe, Libtorch, OpenCV::DNN, to name a few that I've used). This project does what many of them don't, however- it shows you 
 in the up front code what happens in the guts of a neural network, and attempts to show that from a mathematical viewpoint, gradient descent 
 is not as hard to understand as many would make laypersons believe. I will caveat this by saying that programming it from scratch with no 
 assistance from high-level libraries was incredibly tricky at first.
 
 #Requirements
 
 The primary motivation behind this project is that most low-level deep learning libraries are severely underdeveloped as far as deployment is 
 concerned, so I felt the need to implement one with all the bells and whistles of a python program. Additionally, many libraries have the
 annoying requirement to be cross-compiled with CMake. While it's not always super difficult to do, I do think that many developers likely share 
 my frustration here. 
 
 
 Since you don't need CMake files or any of that, to compile the program, simply cd into the DRACO2 folder and run g++ -o draco DRACO.cpp.
 Run with ./draco (or whatever word you chose in the compile command). All you need is g++ set up on your device, and you'll be able to run it.
 You can both train and deploy Draco models in C++ directly.
 While C++ is one of, if not the most challenging of programming languages, the program itself is straightforward to any decent programmer.
 I've abstracted most of the complex tensor calculus logic and low-level memory management in the files architecture.cpp and tensor_calculus.cpp,
 allowing you to build, train, and deploy models in DRACO.cpp with nearly the same level of simplicity as a Python program.
 
 
 TL;DR: Happy Learning!
 

