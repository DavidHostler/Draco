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


float   scalar_dot_product(vector<float> const &A, vector<float> const &B){
    float sum = 0;
    //for(int i = 0; i < A.size(); i++){
    for(int i = 0; i < 2; i++){
        sum = sum + A[i] * B[i];
    }
    return sum;
}

vector<float> ReLU(vector<float> const &Z){
    vector<float> output = {};
    for(int i =0; i < Z.size(); i++){
        if(Z[i] > 0){
            output.push_back(Z[i]);
        } else{
            output.push_back(0);
        }
    }
    return output;
}


vector<float> ReLU_derivative(vector<float> const &Z){
    vector<float> output = {};
    for(int i =0; i < Z.size(); i++){
        if(Z[i] > 0){
            output.push_back(1);
        } else{
            output.push_back(0);
        }
    }
    return output;
}

vector<float> softmax(vector<float> const &a){  
    vector<float> output = {};
    float sum = 0; 
    //float sum = 0;
    for(int j = 0; j < a.size(); j++){
        sum = sum + exp(a[j]); 
    }

    for(int i = 0; i < a.size(); i++){ 
        output.push_back(exp(a[i])/sum);
    }

    return output;
} 


vector<float> crossentropy_loss_deriv(vector<float> const &a){
    vector<float> output = {};
    
    for(int j = 0; j < a.size(); j++){
        output.push_back(1/a[j]);
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


vector<float> LinearTransform2(float W_transpose[2][2],vector<float> const &V){
    int I = 2;
    int J = 2;
    vector<float>    output = {};
    for(int i = 0; i < I; i++){ 
        //resets to empty vector for every row of the matrixs
        vector<float> weight_rows = {}; 
        for(int j = 0; j < J; j++){ 
            weight_rows.push_back(W_transpose[i][j]); 
        }
    //matrix product
    output.push_back(scalar_dot_product(weight_rows,  V));
            
    }  
    return output;
}

vector<float> LinearTransform1(float W_transpose[2][3],vector<float> const &V){
    int I = 2;
    int J = 3;
    vector<float>    output = {};
    for(int i = 0; i < I; i++){ 
        //resets to empty vector for every row of the matrixs
        vector<float> weight_rows = {}; 
        for(int j = 0; j < J; j++){ 
            weight_rows.push_back(W_transpose[i][j]); 
        }
    //matrix product
    output.push_back(scalar_dot_product(weight_rows,  V));
            
    }  
    return output;
}