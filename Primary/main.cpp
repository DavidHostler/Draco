#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <numeric>  
#include <math.h> 
#include "activations.h"
#include "inputs.h"
#include <cstdlib>
#include <ctime>

using namespace std;


class Layers{
    public:
        vector<float> first_layer;
        vector<float> first_activation;
        vector<float> second_layer;
        vector<float> second_activation;

        
};

Layers layers;

class Parameters{
    public:
        //first layer, 2 neurons
        float  W1[2][3];
        vector<float> b1; //Size 2 in Z1 = W1X + b1
        vector<float> Z1;
        //second layer, also 2 neurons
        float W2[2][2];
        vector<float> b2; // size 2 
        vector<float> Z2;


        float W1_T[3][2];
        float W2_T[2][2];


        //Gradients
        vector<float> dZ2;
        float dW2[2][2];
        vector<float> dZ1;
        float dW1[2][3];
        vector<float> db2;
        vector<float> db1 ;
         
}; 

Parameters params;


//First Layer Activation and parameters;
void initialize_first_layer_parameters() { 
    vector<float> matrix_i = {}; 
    int I = 2;
    int  J = 3;
    srand((unsigned int)time(NULL));
    float a = .01; //We have to figure out why this shit keeps causing softmax to output nan if it's not suffic. small.
    cout << "First Weight Matrix" << endl;
    
    for(int i = 0; i < I; i++){ 
        for(int j = 0; j < J; j++){
             
            params.W1[i][j] = float(rand())/float(RAND_MAX) * a; 
            cout << params.W1[i][j] <<  endl;
        
        }
        params.b1.push_back(float(rand())/float(RAND_MAX)*0.1);
        //params.b2.push_back(float(rand())/float(RAND_MAX)*0.1); 
    }  
}    
vector<float> Layer_1(vector<float> const &V){
    int I = 2;
    int J = 3;
    vector<float>    output = {};
    for(int i = 0; i < I; i++){ 
        //resets to empty vector for every row of the matrixs
        vector<float> weight_rows = {}; 
        for(int j = 0; j < J; j++){ 
            weight_rows.push_back(params.W1[i][j]); 
        }
    //matrix product
    output.push_back(scalar_dot_product(weight_rows,  V   ) + params.b1[i]);
            
    }  
    return output;
}



//SeconD Layer Activation and parameters;

 
void initialize_second_layer_parameters() { 
    vector<float> matrix_i = {}; 
    //Dimensions of second layer matrix
    int I = 2;
    int J = 2;
    srand((unsigned int)time(NULL));
    float a = .01; //We have to figure out why this shit keeps causing softmax to output nan if it's not suffic. small.
    for(int i = 0; i < I; i++){ 
        for(int j = 0; j < J; j++){
             
            params.W2[i][j] = float(rand())/float(RAND_MAX) * a;//rand()%20000 / pow(2, 20); //0.25; //randomly initialized. best for 0.25
        }
        params.b2.push_back(float(rand())/float(RAND_MAX)*0.1);
    }  
}    

vector<float> Layer_2(vector<float> const &V){
    int I = 2;
    int J = 2;
    vector<float>    output = {};
    for(int i = 0; i < I; i++){ 
        //resets to empty vector for every row of the matrixs
        vector<float> weight_rows = {}; 
        for(int j = 0; j < J; j++){ 
            weight_rows.push_back(params.W2[i][j]); 
        }
    //matrix product
    output.push_back(scalar_dot_product(weight_rows,  V) + params.b2[i]);
            
    }  
    return output;
}

//Backprops

void W1_transpose(){
    float W1_T[3][2];
    //float transpose[3][2];
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 3; j++){
            params.W1_T[j][i] = params.W1[i][j];
        }
    }
    
}

void W2_transpose(){
    float W2_T[2][2];
    //float transpose[3][2];
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            params.W2_T[j][i] = params.W2[i][j];
        }
    }
    
}

//Call update() to do update weights and biases via grad descent;

void Backpropagation(vector<float> const &x){ //Takes image as input andro
    //Calculate dLoss / dZ2 (output from last hid layer)
    //dZ2 = -1/Y because of logloss derivative

    W1_transpose();
    cout << "Transpose" << endl;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 2; j++){
            cout << params.W1_T[i][j] << endl;
            cout << params.W1[j][i] << endl;

        }
    } 
    //dL / dW2 = dL / dZ2 * X_transpose  , where X_transpose is first activation and 
    //need to take the outer prodcut of the two vectors
    params.dZ2 = crossentropy_loss_deriv(layers.second_activation);
    //float dW2[2][2]; 
    cout << "dW2" << endl;
    cout << "lossder size: " << params.dZ2.size() << endl;
    cout << "First activation size: " << layers.first_activation.size() << endl;
    //Add forloop for the whole batch of vectors and then divide by mini_batch_size
    
    for(int i = 0; i < 2; i++){ //dW2 is 2 * 2
        for(int j = 0; j < 2; j++){ 
            
            params.dW2[i][j] = params.dZ2[i] *  layers.first_activation[j];  
            //      dw2[i][j] = params.dZ2[i] *  layers.first_activation[j];  
            
            //cout << params.W1[i][j] << endl;
            cout << params.dW2[i][j] << endl;

        }
    } 
    //dL / dZ1 = W2_transpose * dL / dZ2 * ReLU_deriv(Z1) 
    W2_transpose(); //initialize W2_transpose 
    params.dZ1 = vector_dot_product(LinearTransform2(params.W2_T, params.dZ2), ReLU_derivative(layers.first_layer));
    cout << "dZ1: " << endl;
    for(int j = 0; j < params.dZ1.size(); j++){
            
           cout << params.dZ1[j] << endl;
           //cout << params.W1[i][j] << endl;
            

        } 
    //dW1  
    cout << "dW1" << endl;
    cout << params.dZ1.size() << endl;
    cout << x.size() << endl;

    
    for(int i = 0; i < 2; i++){
          for(int j = 0; j < 3; j++){
              //outer product
            params.dW1[i][j] =  params.dZ1[i]  *  x[j];
            cout << params.dW1[i][j] << endl; 
            } 
        }

    // dL / db2 = dL / dZ2
    params.db2 = params.dZ2;
    //dL / db1 = dL / Dz1
    params.db1 = params.dZ1;

}

void update_parameters(float learning_rate){
    for(int i =0; i < 2; i++){
                for(int j =0; j < 3; j++){
                params.W1[i][j] = params.W1[i][j] - params.dW1[i][j] * learning_rate; 
            }    
    }

    for(int i =0; i < 2; i++){
                for(int j =0; j < 2; j++){
                params.W2[i][j] = params.W2[i][j] - params.dW2[i][j] * learning_rate; 
            }    
    }


    for(int j = 0; j < params.b2.size(); j++){
                params.b2[j] = params.b2[j] - params.db2[j] * learning_rate; 

    } 
    for(int j = 0; j < params.Z2.size(); j++){
                params.Z2[j] = params.Z2[j] - params.dZ2[j] * learning_rate; 


    }
 
    for(int j = 0; j < params.b1.size(); j++){
                params.b1[j] = params.b1[j] - params.db1[j] * learning_rate; 
    }     

    for(int j = 0; j < params.Z1.size(); j++){
                params.Z1[j] = params.Z1[j] - params.dZ1[j] * learning_rate; 

    }
}
 

int main(){
    //Initialize random weights and biases
    initialize_first_layer_parameters();
    initialize_second_layer_parameters();
    float inputs[3][3] = {{0.1,0.2,0.3 }, {0.3,0.1,0.1},{0.5, 0.9, 0.1}}; //3 sets of 3 element vectors 
    vector< float> labels = {1,1};
    //Forward propagation;
    float  loss = 0; 
    for(int epochs = 0; epochs < 100; epochs++){
        float loss_table[3][2]; //data structure to make summing losses moar convenient, i.e. LUT
        //#cols = number of vectors in batch size, #rows = size of softmax layer
        for(int k = 0; k < 3; k++){ //for each vector in the minibatch
            vector<float> loss_vector;

            //vector<float> x = input_vector(inputs, k);
            //vector<float> y = {0.3,0.1,0.1};
            //vector<float> z = {0.5, 0.9, 0.1};
            
            layers.first_layer =  Layer_1(input_vector(inputs, k)); //Z1
            layers.first_activation = ReLU(layers.first_layer);
            layers.second_layer = Layer_2(layers.first_activation);
            layers.second_activation = softmax(layers.second_layer); //segfault if the 2nd layer isn't initialized!
            //cout << "Matrix product W1 * X: " << endl;
            cout << "Outputs of last layer used: " << endl;
            cout << layers.first_activation.size();
            loss_vector =   crossentropy_loss(layers.second_activation);

            for(int j = 0; j < layers.second_activation.size(); j++){ 

                //cout << first_layer[j] << endl;
                //cout << first_activation[j] << endl;

                //cout << second_layer[j] << endl;

                cout << layers.second_activation[j] << endl;

                }
                //Basically onehot encoding
            
            //do this for each of the vectors in the batch
            loss = loss + scalar_dot_product(loss_vector,labels);



        }
        cout << "Loss: " << loss << endl;
        //Backpropagation! 
        //This applies stochastic gradient descent/minibatches to the idea
        float dw1[2][3];
        float dz1[2];
        float dw2[2][2];
        float dz2[2];
        for(int k = 0; k < 3; k++ ){

            Backpropagation(input_vector(inputs, k));
            for(int i = 0; i < 2; i++){
                for(int j = 0; j < 3; j++){
                        dw1[i][j] = dw1[i][j] + params.dW1[i][j]; 
                }
            }
           
            for(int j = 0; j < 2; j++){
                    dz1[j] = dz1[j] + params.dZ1[j]; 
            } 

            for(int i = 0; i < 2; i++){
                for(int j = 0; j < 2; j++){
                        dw2[i][j] = dw2[i][j] + params.dW2[i][j]; 
                }  
            }
            for(int j = 0; j < 2; j++){
                    dz2[j] = dz2[j] + params.dZ2[j]; 
            } 
            //set params = 0;
            for(int i = 0; i < 2; i++){
                for(int j = 0; j < 3; j++){
                        params.dW1[i][j] = 0; 
                }   
            }
            
        
        //Averages the weights; do this for all gradient parameters;
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 3; j++){
                params.dW1[i][j] = dw1[i][j]/3 ; // 3 is the batch_size
            }   
        
        } 
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 2; j++){
                params.dW2[i][j] = dw2[i][j]/3 ; // 3 is the batch_size
            }    
        } 

        for(int j = 0; j < 2; j++){
                params.dZ1[j] = dz1[j]/3 ; // 3 is the batch_size
            } 
        for(int j = 0; j < 2; j++){
                params.dZ2[j] = dz2[j]/3 ; // 3 is the batch_size
            }
 
        }
        
        update_parameters(0.00001); //learning rate = 0.01
        

    }


    for(int j = 0; j < layers.second_activation.size(); j++){  

                cout << "Predictions: " << j << " "  << layers.second_activation[j] << endl;
                

        }
    
}
