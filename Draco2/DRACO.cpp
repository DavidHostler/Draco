#include <iostream>
#include <stdio.h>
#include "architecture.cpp"
// #include "tensor_calculus.cpp"
#include <typeinfo>
#include <cmath>
using namespace std;

float ** generate_training_input(int vector_size, int index_dataframe){
        float ** dataset = new float * [2];
        dataset[0] = new float[vector_size];
        dataset[1] = new float[1];
        float * X_train_i = new float;
        float* y_train_i = new float;
        y_train_i[0] = 1.0;
        int exponent;
        if(index_dataframe % 2 == 0){
            y_train_i[0] = float(0.0);
            exponent = 2; //
        }
        else{
            y_train_i[0] = float(1.0);
            exponent = 1;
        }
        //Generate random normalised vectors 
        float one = 1.0;
        for(int i = 0; i < vector_size; i++){
            //Even numbers have 0 as their modulus 2, so the result will be * + 1
            float a = pow(0.95, i) * pow(-1, exponent); //This introduces randomness 
            srand((unsigned int)time(NULL));
            X_train_i[i] = float(rand()) / float((RAND_MAX)) * a;
            
        }
        
        dataset[0] = X_train_i;
        dataset[1] = y_train_i;

        return dataset;
    }

int main()
{
    //Input data in the form of floating point static arrays

    //Generate Training data
    //Input arrays of size two containing the training vector and the ground truth-label 
    // float *** generate_training_data(int N){
    //     //Every odd vector will have ground truth of 1, even would be 0 for all of these
         
         
    // }
    float *** training_data = new float ** [1000];
    for(int k = 0; k < 1000; k++){
        
        training_data[k] = new float * [2];
    }
    
    //Sample training data

    float ** data = generate_training_input(10, 5);
    // cout << "X_train_i" << endl;
    // print_vector(data[0], 10);
    // print_vector(data[1], 1);
    int counter = 0;
    for(int m = 0; m < 1000; m++){
        data = generate_training_input(10, m);
        training_data[m] = data;
        counter++;
    }
    // cout << "The number of data entries added to our dataset: " << counter << endl;

    

    // float value = random_init(); 
    // cout << "RAND: " << value << endl;
    float input[3] = {0.1, 0.341, 0.985};
    float input_negative[3] = {-0.243, -0.90815, -0.351};
    // float y_train_i = 1.0;
    float y_ = 1.0;

    //Initialize Draco Model
    Model model;
    Dense * new_layer = new Dense;
    //Initialize input layer 
    
    new_layer->initialize_weights(5,3,false);
    model.input_layer = new_layer;
    model.input_layer->activation_function = "relu"; //Do this manually because the forward pass requires this value
                                                     //to be prestored in the data structure
    model.add_layer(2, 5, "relu", false);
    model.add_layer(1, 2, "sigmoid", false); //Boolean signals not to display parameters of layer in terminal 

    model.train(10, 10, 0.001, training_data, 1000);

    float * pred_positive = model.predict(input);
    cout << "Output prediction after being trained for positive input vector" << endl;
    print_vector(pred_positive, model.output_layer->m); 

    float * pred_negative = model.predict(input_negative);
    cout << "Output prediction after being trained for positive input vector" << endl;
    print_vector(pred_negative, model.output_layer->m); 
    //Training loop
    // model.forward_pass(input, y_); //Run the forward pass!
    // model.backpropagation(0.001);
    // int EPOCHS = 10;
    // //Lets train a batch of these at once!
    // float * current_input_vector = new float;
    // float current_y; 
    // int batch_size = 10;
    // int size_of_dataset = counter;
    // int num_batches = int(size_of_dataset/batch_size);
    // int start = 0;
    // int end = 10;
    // for(int e = 0; e < EPOCHS; e++){

    //         for(int index = start; index < end; index++){//counter is the size of our dataset; 
    //                                                     //in this case, counter is our batchsize
    //             current_input_vector = training_data[index][0]; 
    //             current_y = training_data[index][1][0]; //The index of data[1] is itself a 1 x 1 array so use [0] to get float
    //             model.forward_pass(current_input_vector, current_y, true); //Run the forward pass!
    //             model.backpropagation(0.001);
    //         }
         

    //         cout << "Completed Epoch " << e << endl;
    //         cout << "Training loss is: " << model.total_loss << endl;
    //         model.total_loss = 0.0; //Reset the loss function after each epoch otherwise you'll just be adding successive 
    //                                 //losses, and it won't be able to decrease over time.   
    //         end += batch_size;
    //         start += batch_size;

    // }
    

    /* 
     *
     * 
     * Get the values of each layer's weights and biases  
     * 
    print_array(model.input_layer->weights, model.input_layer->m, model.input_layer->n);
    print_array(model.input_layer->next->weights, model.input_layer->next->m, model.input_layer->next->n);
    print_array(model.input_layer->next->next->weights, model.input_layer->next->next->m, model.input_layer->next->next->n);
    *
    */
    
}