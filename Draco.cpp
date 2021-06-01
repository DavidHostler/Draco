#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <numeric>  
#include <math.h> 
using namespace std;
 
//matrix.cpp
class Layers{
    public: 
        //outputs of each hidden layer specified;
        vector<float> layer1;
        vector<float> layer2;
        vector<float> layer3; 
};

Layers layers;


class Parameters{
    public:
        float dot_product;
        float W1[100][784];
        float W2[100][784];
        vector<float>b1 = {};
        vector<float>b2 = {};
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
void dot(vector <float> const &a,vector <float> const &b){
                params.dot_product = 0;   
                for(int i = 0; i < a.size(); i++){ 
                    params.dot_product = params.dot_product + a[i] * b[i] ; 
                }  
        } 

//Generates the matrices and vectors (Initialize parameters!)
 void initialize_parameters(int I, int  J) { 
    vector<float> matrix_i = {}; 
    for(int i = 0; i < I; i++){ 
        for(int j = 0; j < J; j++){
             
            params.W1[i][j] = rand() / pow(10, 10) - 0.105; //randomly initialized. best for 0.105
            params.W2[i][j] = rand() / pow(10, 10) - 0.105; //randomly initialized. best for 0.105 
        }
        params.b1.push_back(rand() / pow(10, 10));
        params.b2.push_back(rand() / pow(10, 10)); 
    }  
}       

//input to activation function layered 
vector<float> ReLU(vector <float> const &a ){ 
        vector<float> out;
        for(int i =0; i < a.size(); i++){
            if(a[i] < 0){  
            out.push_back(0);
            } else{ 
                out.push_back(a[i]);
            }  
        } 
        return out; 
    }


vector<float>  layer(string activation, vector <float> const  &inputs){
        vector<float> layer;
        if(activation == "ReLU"){
            layer = ReLU(inputs);
        }else {
            layer = {0};    //will expand to include moar activation functions down the road!
        } 
        return layer;
        
    }
  
   
 //W1 x X + b1
vector<float> Linear_Transform(float matrix[100][784], vector <float> const  &X,vector <float> const  &b1){ 
        int c = 0;
        vector<float> Z ={};
        for(int i = 0; i < 100; i++){    //m = 100 rows
            vector<float> matrix_i = {};  
            for(int j =0; j < 784; j++){     //n = 784 colms;
                //matrix_i.push_back(params.W1[i][j]);  
                matrix_i.push_back(matrix[i][j]);  
                } 
                //Row vector dot product with X;
                dot(matrix_i, X);
                //cout << params.dot_product << endl; 
                Z.push_back(params.dot_product + b1[i]);
                c++;
            }
            return Z;
    }



int main(){
      
    vector<float> X = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 85, 85, 85, 85, 85, 85, 85, 85, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 128, 168, 250, 250, 250, 252, 250, 250, 250, 250, 231, 127, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 168, 237, 252, 250, 250, 250, 250, 252, 250, 250, 250, 250, 252, 250, 209, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 250, 250, 252, 250, 250, 250, 250, 252, 250, 250, 250, 250, 252, 250, 250, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 113, 252, 252, 252, 247, 210, 210, 210, 210, 177, 0, 0, 0, 0, 43, 252, 252, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 250, 250, 250, 250, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 194, 250, 138, 14, 0, 0, 0, 0, 0, 0, 0, 0, 43, 250, 250, 250, 250, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 250, 250, 41, 0, 0, 0, 0, 0, 0, 0, 0, 43, 250, 250, 137, 83, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 167, 250, 41, 0, 0, 0, 0, 0, 0, 0, 0, 219, 250, 144, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 250, 217, 0, 0, 0, 0, 0, 0, 0, 0, 254, 238, 105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 148, 252, 252, 0, 0, 0, 0, 0, 0, 0, 0, 252, 166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 140, 250, 250, 179, 0, 0, 0, 0, 0, 0, 0, 0, 252, 208, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 127, 252, 250, 250, 250, 41, 0, 0, 0, 0, 0, 0, 0, 0, 252, 250, 209, 56, 0, 0, 0, 0, 0, 141, 170, 168, 168, 223, 250, 252, 250, 250, 137, 14, 0, 0, 0, 0, 0, 0, 0, 0, 252, 250, 250, 223, 210, 212, 210, 210, 210, 244, 252, 250, 250, 250, 250, 252, 250, 144, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 252, 252, 252, 252, 254, 252, 252, 252, 252, 255, 252, 252, 252, 217, 177, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 166, 208, 250, 250, 252, 250, 250, 250, 250, 238, 166, 166, 166, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 125, 125, 146, 250, 250, 165, 125, 105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 83, 83, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    //100 neurons for the first hidden layer, 784 pixel image input is used!
    initialize_parameters(100,784);
       
    //Performs the above commented math operation of multiplying W1 * X + b1 ;
    vector<float> Z = Linear_Transform(params.W1, X, params.b1); //Output result gives Z, input vector to activation function first lad 
    layers.layer1 = layer("ReLU", Z); //Assign to Layers class; 
    vector<float> A1 = Linear_Transform(params.W2, layers.layer1, params.b1); 
    layers.layer2 = layer("ReLU", A1);


    //Write the SoftMax function and forwardprop is completed!

    //Here is what the output of the first layer looks like:
    for(int i = 0; i < layers.layer2.size(); i++){
        //first_layer.push_back(ReLU(Z)[i]); 
        cout << layers.layer2[i] << endl;
    }
    
     
    
     
      
    return 0;
}