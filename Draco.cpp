#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <numeric>  
#include <math.h> 
#include "functions.h"
#include <cstdlib>
#include <ctime>

using namespace std;
 
//matrix.cpp
class Layers{
    public: 
        //activation functions of each hidden layer specified;
        vector<float> layer1;
        vector<float> layer2;
        vector<float> layer3; 
};

Layers layers;


class Parameters{
    public:
        float dot_product;
        float W1[10][784];
        float W2[10][10];
        float dW1[10][784], dW2[10][10]; //gradients 
        float W2_transpose[10][10]; 
        float W1_transpose[784][10]; 
        vector<float> dZ1;
        vector<float> dZ2;
        vector<float>b1 = {};
        vector<float>b2 = {};
        vector<float> db1 = {};
        vector<float> db2 = {}; // gradients
        vector<float> A1;
};
//Instantiate object of the class;
Parameters params;


int derivative(int y){
            float h = 0.001;
            float y0, y1; //y1 = y(x + h), y0 = y(x)
            float derivative = (y1 - y0)/h;
            return derivative;
        }
 
//Takes Y training data and ouputs array of 1ss
vector<int> one_hot_y(vector <int> const &Y){
            vector<int> one_hot_y ;
            for(int i = 0; i <= Y.size(); i++){
                one_hot_y.push_back(i);
                //return one_hot_y[i] ;
                
            } 
            return one_hot_y;
        }        

//Generates the dot product;
float dot(vector <float> const &a,vector <float> const &b){
                params.dot_product = 0;   
                for(int i = 0; i < a.size(); i++){ 
                    params.dot_product = params.dot_product + a[i] * b[i] ; 
                }  
            return params.dot_product;
        } 

//Generates the matrices and vectors (Initialize parameters!)
 void initialize_parameters(int I, int  J) { 
    vector<float> matrix_i = {}; 
    srand((unsigned int)time(NULL));
    float a = .000000000000001; //We have to figure out why this shit keeps causing softmax to output nan if it's not suffic. small.
    for(int i = 0; i < I; i++){ 
        for(int j = 0; j < J; j++){
             
            params.W1[i][j] = float(rand())/float(RAND_MAX) * a;//rand()%100000 / pow(10, 20); //0.105; //randomly initialized. best for 0.105
            params.W2[i][j] = float(rand())/float(RAND_MAX) * a; //rand()%100000 / pow(10, 20); //0.105; //randomly initialized. best for 0.105 
        }
        params.b1.push_back(float(rand())/float(RAND_MAX)*0.1);
        params.b2.push_back(float(rand())/float(RAND_MAX)*0.1); 
    }  
}         
 
 vector<float> Linear_Transform_Forward_1(float matrix[10][784], vector <float> const  &X,vector <float> const  &b1){ 
        int c = 0;
        vector<float> Z ={};
        for(int i = 0; i < 10; i++){         //m = 10 rows
            vector<float> matrix_i = {};  
            for(int j =0; j < 784; j++){     //n = 784 colms;
                //matrix_i.push_back(params.W1[i][j]);  
                matrix_i.push_back(matrix[i][j]);  
                } 
                //Row vector dot product with X;
                //dot(matrix_i, X); 
                Z.push_back(dot(matrix_i, X) + b1[i]);

                c++;
            }
            return Z;
    }  



vector<float> Linear_Transform_Forward_2(float matrix[10][10], vector <float> const  &X,vector <float> const  &b1){ 
        int c = 0;
        vector<float> Z ={};
        for(int i = 0; i < 10; i++){         //m = 10 rows
            vector<float> matrix_i = {};  
            for(int j =0; j < 784; j++){     //n = 784 colms;
                //matrix_i.push_back(params.W1[i][j]);  
                matrix_i.push_back(matrix[i][j]);  
                } 
                //Row vector dot product with X;
                //dot(matrix_i, X); 
                Z.push_back(dot(matrix_i, X) + b1[i]);

                c++;
            }
            return Z;
    }  
//For backprop to calculate dZ1
vector<float> Linear_Transform_Backward(float matrix[784][10], vector <float> const  &X){ 
        int c = 0;
        vector<float> Z ={};
        for(int i = 0; i < 10; i++){         //m = 10 rows
            vector<float> matrix_i = {};  
            for(int j =0; j < 784; j++){     //n = 784 colms;
                //matrix_i.push_back(params.W1[i][j]);  
                matrix_i.push_back(matrix[i][j]);  
                } 
                //Row vector dot product with X;
                //dot(matrix_i, X); 
                Z.push_back(dot(matrix_i, X));

                c++;
            }
            return Z;
    }  
//float grad_w[10][784];
//Rename backpropagation d
void matrix_outer_product_1(vector<float> const &A, vector<float> const &X, float dW[10][784]){  //dLdY  = SOFTMAX
    //Initialize dW2  = softmax_output * ReLU T
    //First set of updates;
    for(int i =0; i < A.size(); i++){
            for(int j =0; j < X.size(); j++){ 
                  vector<float> dZ2 = A; //Vector update
                  dW[j][i] =  A[i] * X[j];  //Dfirst weight update, outer product 
            } 
        }  
}

void matrix_outer_product_2(vector<float> const &A, vector<float> const &X, float dW[10][10]){  //dLdY  = SOFTMAX
    //Initialize dW2  = softmax_output * ReLU T
    //First set of updates;
    for(int i =0; i < A.size(); i++){
            for(int j =0; j < X.size(); j++){ 
                  vector<float> dZ2 = A; //Vector update
                  dW[j][i] =  A[i] * X[j];  //Dfirst weight update, outer product 
            } 
        }  
}

void vector_outer_product(vector<float> const &A, vector<float> const &X,  vector<float> const &dZ){  //dLdY  = SOFTMAX
    //Initialize dW2  = softmax_output * ReLU T
    //First set of updates;
    for(int i =0; i < A.size(); i++){
            for(int j =0; j < X.size(); j++){ 
                  vector<float> dZ2 = A; //Vector update
                  //dZ[j][i] =  A[i] * X[j];  //Dfirst weight update, outer product 
            } 
        }  
}

 
void backpropagation(vector<float> const &X){  //dLdY  = SOFTMAX
    //Initialize dW2  = softmax_output * ReLU T
    //First set of updates;
    params.dZ2 = dLdY(layers.layer2);
    matrix_outer_product_2(params.dZ2, X, params.dW2); //updates dW2
    //matrix_outer_product(params.W2, dLdY(layers.layer2), params.dZ1));//updates dZ1; needs a version for matrix inputs
    vector<float> Z1 = Linear_Transform_Forward_1(params.W1, X,params.b1); 
    //Update dE/dZ1 
    Transpose(params.W2, params.W2_transpose);
    params.dZ1 = vector_dot_product(Linear_Transform_Backward(params.W2_transpose, params.dZ2), ReLU_derivative(Z1));
    matrix_outer_product_1(params.dZ1, X, params.dW1); // dW1  
} 
//Loop thru dis to update weights and biases! 
//update_parameters(float dW1, float dW2,float db1, float db2,  float learning_rate)
void update_parameters(float learning_rate){ //dW = dL/dW shorthand
    for(int i =0; i < 10; i++){
                for(int j =0; j < 784; j++){
                params.W1[i][j] = params.W1[i][j] - params.dW1[i][j] * learning_rate;
                params.W2[i][j] = params.W2[i][j] - params.dW2[i][j] * learning_rate; 
            }  
            params.b1[i] = params.b1[i] - params.db1[i] * learning_rate;
            params.b2[i] = params.b2[i] - params.db2[i] * learning_rate; 
            
    }
}

//WORKING CODE USED

void forward(vector<float> const &X){

    //10 neurons for the first hidden layer, 784 pixel image input is used!
    initialize_parameters(10,784);  
    vector<float> Z = Linear_Transform_Forward_1(params.W1, X, params.b1); //Output result gives Z, input vector to activation function first lad 
    layers.layer1 = layer("ReLU", Z); //Assign to Layers class; 
    vector<float> A1 = Linear_Transform_Forward_2(params.W2, layers.layer1, params.b1);  
    layers.layer2 = layer("softmax", A1);  

 

    //Here is what the output of the first layer looks like:
    
    float total_loss = 0;
    cout << "X size: " << X.size() << ", Z size: " << Z.size() << ", LAYERSIZE: " << layers.layer1.size() << endl;
    for(int i = 0; i < layers.layer2.size(); i++){
        //first_layer.push_back(ReLU(Z)[i]); 
        float total_loss = total_loss + loss(layers.layer2)[i]; 
        cout << "Loss contrib: " << loss(layers.layer2)[i] << endl;
        cout     << "Z: " << Z[i] << ", layer 1: " << layers.layer1[i] << ", layer 2: " << layers.layer2[i] << endl;
        cout << "W1:  " <<  params.W1[i][0] << ", W2: " << params.W2[i][0]<< " rand: " << rand() << endl;  
        //total = total + layers.layer2[i];
        //cout << layers.layer1[i] << endl;
 
    } 
        cout << "Total loss: " << total_loss;
    //cout << "Total probability: " << total;
    

     
}
  


int main(){
      
    vector<float> X = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 85, 85, 85, 85, 85, 85, 85, 85, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 128, 168, 250, 250, 250, 252, 250, 250, 250, 250, 231, 127, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 168, 237, 252, 250, 250, 250, 250, 252, 250, 250, 250, 250, 252, 250, 209, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 250, 250, 252, 250, 250, 250, 250, 252, 250, 250, 250, 250, 252, 250, 250, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 113, 252, 252, 252, 247, 210, 210, 210, 210, 177, 0, 0, 0, 0, 43, 252, 252, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 250, 250, 250, 250, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 194, 250, 138, 14, 0, 0, 0, 0, 0, 0, 0, 0, 43, 250, 250, 250, 250, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 250, 250, 41, 0, 0, 0, 0, 0, 0, 0, 0, 43, 250, 250, 137, 83, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 167, 250, 41, 0, 0, 0, 0, 0, 0, 0, 0, 219, 250, 144, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 250, 217, 0, 0, 0, 0, 0, 0, 0, 0, 254, 238, 105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 148, 252, 252, 0, 0, 0, 0, 0, 0, 0, 0, 252, 166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 140, 250, 250, 179, 0, 0, 0, 0, 0, 0, 0, 0, 252, 208, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 127, 252, 250, 250, 250, 41, 0, 0, 0, 0, 0, 0, 0, 0, 252, 250, 209, 56, 0, 0, 0, 0, 0, 141, 170, 168, 168, 223, 250, 252, 250, 250, 137, 14, 0, 0, 0, 0, 0, 0, 0, 0, 252, 250, 250, 223, 210, 212, 210, 210, 210, 244, 252, 250, 250, 250, 250, 252, 250, 144, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 252, 252, 252, 252, 254, 252, 252, 252, 252, 255, 252, 252, 252, 217, 177, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 166, 208, 250, 250, 252, 250, 250, 250, 250, 238, 166, 166, 166, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 125, 125, 146, 250, 250, 165, 125, 105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 83, 83, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    float total_loss = 0;
    for(int epochs = 0      ; epochs < 1; epochs++){

        forward(X); //forloop thru images used for the inputs

        backpropagation(X); //Update differential of the weights
    
        update_parameters(0.01); 
        

    }
    
    //Successfully computed weight gradients!
    //cout << params.dW1[1][1] << endl;
    //cout << params.db1[1] << endl; //Segfault err


    //float dW2[10][10]

    
     
      
    return 0;
    
}
