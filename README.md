Hello, and welcome to the Draco library for deployment of 
deep neural networks. 
Developed by David Hostler (me) during 2021, this project was designed to help 
me develop a deeper understanding of the operations under the hood in deep learning models. 

This framework does have training built in via C++, 
but in the future for simplicity, its primary use may be to act as a 
deployment API for models in production. 

One day you'll train a model in Pytorch, load the state dictionary to 
Draco, and use the Draco API under the hood to power your (hopefully friendly!)
totally-not-a-terminator-robot.


I chose C++ to build the library for the following reasons"
    I. Unparalleled speed of execution, which potentially makes the framework 
    excellent for robotics and embedded-systems applications.
    II. It gives me direct access to memory allocation, which is a superpower 
    that Python developers like myself often take for granted. It additionally helps catch errors when the size of an input vector to a hidden layer doesn't have the same number of dimensions as the columnspace of the layer matrix (The notorious "Segmentation Fault" error saves me from a lot of silent failures.)
    III. Built in concurrency which native Python generally doesn't have. Even in deployment, many large deep learning models rely on GPU's for inference
    in order to parallelize matrix operations. While still optimal, now you can run some operations multithreaded on CPU.  
    IV. No added dependencies or "bloatware". Simply compile and run the program. 

There is much work to be done, and I would greatly appreciate the help of any other developers to give me a hand in optimizing the project.
Right now, I have a small wishlist for Draco:

I. Currently, I have only implemented a Dense (Linear) layer class.
If this framework is to be truly robust, it must have the additional layers 
available to developers:
    * Dropout
    * Conv2D (Images)
    * Conv1D
    * MaxPooling2D
    * Embedding (Important for NLP tasks!)

II. Certain operations appear redundant. 
For instance, there exists in tensor_calculus.cpp a function called 
"einstein_sum", which does not appear to be used in any uncommented code 
throughout the project. 


III. Shaders/Concurrency- I promised the people concurrency and parallelizability. If someone wants to give me a hand writing shaders, I'd be thrilled to have you contribute!