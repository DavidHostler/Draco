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

float inputs[3][3] ; //3 sets of 3 element vectors
      

void generate_inputs(){
    float a = 0;
    float b = 1;
    //convert array into vectors;
       for(int i =0; i <3; i++){ 
            for(int j =0; j <3; j++){
                
                inputs[i][j] =  (a * b)  / 100;
                a++;
                b++;
                //std::cout << inputs[i][j];
        }

    }       
}

//converts rows of input array into vectors
vector<float> input_vector(float inputs[3][3], int index){
    vector<float> output;
    for(int j =0; j <3; j++){ 
            output.push_back(inputs[index][j]); 
    }       
    return output;
}


 