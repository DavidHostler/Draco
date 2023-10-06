#include <iostream>
using namespace std;

float ** heap_array(int m, int n){//Allocate 2D array of floats on the heap memory 
    float ** result = new float * [m];

    for(int i = 0; i < m; i++){
         result[i] = new float;
    }
    return result;
}

void print_array(float ** matrix, int rows, int cols){
    for(int i = 0; i < rows; i++){
        cout << "Row: " << i << " " << endl;
        for(int j = 0; j < cols; j++){
            cout << matrix[i][j] << endl;
        }
    }
}

void print_vector(float * vec, int m){
    cout << "The components of this vector are..." << endl;
    for(int i = 0; i < m; i++){
        cout << vec[i] << endl;
    }
}

float ** tensor_product(float * u, float * v, int m, int n){
    //u is a column vector with m rows undergoing an outer product with row vector v
    //with n cols
    //Initialize the output data structure with a pointer
    float ** product = heap_array(m, n);

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            
            product[i][j] = u[i] * v[j];
        }
    }
    
    return product;

}

float inner_product(float * u, float * v, int m){
    float result = 0;
    for(int i = 0; i < m; i++){
            result += u[i] * v[i];
    }

    return result;
}


// Deprecate? This could possibly be replaced via diagonal matrix multiplication.
float * einstein_sum(float * u, float * v, int m){
    float * result = new float;
    for(int i = 0; i < m; i++){
        result[i] = u[i] * v[i]; //Elementwise multiplication as per einstein summation notation, ui vi
    }
    return result;
}

//Takes the derivative of the output vector of an activation fucntion wrt 
//It's inputs. Used in chain rule derivatives; output is a 1D array
float * vector_derivative_of_loss(float * u, int m, float x0, float(* func)(float x0,float x)){
    float h = 0.001;
    float func1, func2;
    float * gradient = new float;
    for(int i = 0; i < m; i++){
        func1 = func(x0, u[i]);
        func2 = func(x0, u[i] + h);
        gradient[i] = (func2 - func1) / h;
    }
    
    return gradient;

    
}


float * vector_derivative_of_activation(float * u, int m, float(* func)(float x)){
    float h = 0.001;
    float func1, func2;
    float * gradient = new float;
    for(int i = 0; i < m; i++){
        func1 = func(u[i]);
        func2 = func(u[i] + h);
        gradient[i] = (func2 - func1) / h;
    }
    
    return gradient;

    
}


float ** transpose(float ** matrix, int m, int n){
    float ** T = heap_array(n, m);//n x m tensor
     
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++)  {

            T[i][j] = matrix[j][i];

            
        }
    }
    
    return T;
}


//This is a poorly named function.
//It is the linear transformation of the transpose of the weights of a given dense layer 
//operating on the chain rule derivative from the next layer during backpropagation, and should
//have been identified as such
float * matrix_outer_product(float ** matrix, float * vector,  int rows, int cols){
    float ** column_vector_matrix = transpose(matrix, rows, cols);
    

    float * result = new float;
    //resultant vector will have size == cols
    for(int j = 0; j < cols; j++){

        result[j]   =   inner_product(vector, column_vector_matrix[j], rows);
        // cout << "matrix outer product elements: " <<  result[j] << endl;
        
    }
    return result;

}


float ** add_matrices(float ** A, float ** B, int rows, int cols){
    float ** result = heap_array(rows, cols);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}


float * add_vectors(float * u, float * v, int m){
    float * sum = new float;
    for(int i = 0; i < m; i++){
        
        sum[i] = u[i] + v[i];
    }
    return sum;
}


float ** multiply_matrix_by_scalar(float ** dW, float learning_rate, int m, int n){
    float ** result = heap_array(m, n);
    for(int i = 0; i < m; i++)  {
        for(int j = 0; j < n; j++) {
            result[i][j] = dW[i][j] * learning_rate;
        }
    }

    return result;
}


float * multiply_vector_by_scalar(float * vec, float learning_rate, int m){
    float * result = new float;
    for(int i = 0; i < m; i++){
        result[i] = vec[i] * learning_rate;
    }
    return result;
}


//Some additional functions


float loss_i(float y, float y_pred)
{
    return y * log(y_pred) + (1 - y) * log(1 - y_pred);
}
// int main(){
//     float ** weight = heap_array(2,2);
//     weight[0][0] = 1.0;
//     weight[0][1] = 1.0;
//     weight[1][0] = 0.0;
//     weight[1][1] = 0.0;

     
//     float ** t_ = transpose(weight, 2, 2);
//     print_array(weight, 2, 2);
//     print_array(t_, 2, 2);

//     float V[2] = {1.0, 2.0};
    
//     float * result = matrix_outer_product(t_, V, 2, 2);
//     for(int i = 0; i < 2; i++){
//         cout << result[i] << endl;
//     }
    
// }



/*
// void backpropagate(float learning_rate){ //Update weights and biases of previous layers...
    //     if(this->prev == nullptr){
    //         return; //There is no previous layer so end the backpropagation leg of the algorithm!
    //     }
        
    //     this->chain_deriv = matrix_outer_product(this->weights, this->chain_deriv, this->m, this->n);
    //     // Update d_activation to now deal with the derivatives of the previous activation
    //     if(this->activation_function == "relu"){
    //         this->d_activation = vector_derivative_of_activation(this->prev->linear_transformation, this->prev->m, relu);

    //     }else if(this->activation_function == "sigmoid"){
    //         this->d_activation = vector_derivative_of_activation(this->prev->linear_transformation, this->prev->m, sigmoid);

    //     }
    //     else{
    //         cout << "Previous hidden layer does not have a valid activation function. Look into it maybe?" << endl;
    //         return;
    //     }
    //     this->chain_deriv = einstein_sum(this->chain_deriv, this->d_activation, this->prev->m);
        
    //     float **dW = tensor_product(this->chain_deriv, this->input_vector, this->prev->m, this->prev->n);

    //     float *db = chain_deriv;

    //     //Multiply gradients by a timestep`
    //     learning_rate *= -1.0;
    //     dW = multiply_matrix_by_scalar(dW, learning_rate, this->m, this->n);
    //     db = multiply_vector_by_scalar(db, learning_rate, this->m);
        
    //     //Update the weights and bias
    //     this->weights = add_matrices(this->weights, dW,  this->m, this->n);
    //     this->bias = add_vectors(this->bias, db, this->m);

    // }

*/