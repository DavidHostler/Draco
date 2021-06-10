# Draco
Open source low-level deep learning framework for C++ and Julia.


A full single hidden layer neural network!
Shortly I will add backpropagation and gradient descent, and work on using
Conv2D nets and dropout in additional code. This is a work in progress!

Draco_cpp is a brute-force approach to getting around build issues, and is the more extreme approach 
as it is written entirely in C++; however, if DracoJL can be embedded directly into somebody's C/C++
program, then engineers less comfortable with low-level code should be able to use it by simply calling it
into the main.cpp file in the future.

The ActorNetwork.jl file is a purely Julia implementation of a set of fully connected layers used in the deep deterministic
policy gradient algorithm, an example of which can be found on the Keras website.
Another version was done in C++, but I have realized not too many developers want to sift through such a monstrous amount of code
to manually tweak hyperparameters. If I can condense that version, then I'll make that one accessible as well.
