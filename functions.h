#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <numeric>  
#include <math.h>  
#include <cstdlib>
#include <ctime> 
using namespace std;


vector<float> softmax(vector<float> const &a){  
    vector<float> output = {};
    float sum = 0; 
    //float sum = 0;
    for(int j = 0; j < 10; j++){
        sum = sum + exp(a[j]); 
    }

    for(int i = 0; i < 10; i++){ 
        output.push_back(exp(a[i])/sum);
    }

    return output;
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
        }else if(activation == "softmax") {
            layer = softmax(inputs);    //will expand to include moar activation functions down the road!
        } 
        else{
            layer = {0};
        }
        return layer;
        
    }

//Crossentropy loss for image classifiers
vector  <float> loss(vector<float> const &V){
    vector<float> out;
    for(int i =0; i < V.size(); i++){
        out.push_back(-log(V[i]));    
    }
     return out;
}

//Derivative of crossentropy loss; takes output of softmax layer as independent variable
vector<float> dLdY(vector<float> const &Y){
    vector<float> out;
    for(int i = 0; i < Y.size(); i++){
         out.push_back(-1/Y[i]);
    }

    return out;
}

vector<float> ReLU_derivative(vector<float> const &Z){
    vector<float> output = {};
    for(int i = 0; i < Z.size(); i++ ){
       if(Z[i] > 0){
       output.push_back(1.);  
       } else{
           output.push_back(0.);
       }
    }

    return output;
}

 


vector<float> vector_dot_product(vector<float> const &A,vector<float> const &B  ){
    vector<float> output;
    for(int i = 0; i <A.size(); i++){ 

        output.push_back(A[i]*B[i]);

    }

    return output;
}


void Transpose (float mat[10][10], float transpose[10][10]){   
    for(int i = 0; i < 2; i++ ) {
        for(int j = 0; j < 2; j++ ){
                transpose[i][j] = mat[j][i]; 
        }
    } 
}



