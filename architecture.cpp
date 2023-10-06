#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <math.h>
#include "tensor_calculus.cpp"
using namespace std;



float random_init()
{ //Initialize random values to seed initial weights and biases
    srand((unsigned int)time(NULL));
    float a = 1.0;
    float random_float = float(rand()) / float((RAND_MAX)) * a;
    return random_float;
}

float relu(float x){
    if(x > 0.0){
        return x;
    }
    else{
        return 0.0;
    }
}

float sigmoid(float x){

    return 1./(1. + exp(-1 * x));
}
// WARNING- we can deprecate this class later in favour 
// of a more general layer class following a Factory Design Pattern.
// We could tweak some instances of the class to be non-Dense, 
// such as a Convolution, etc.
//Dense hidden layer
class Dense
{
public:
    Dense* current;
    Dense* next;
    Dense* prev;
    int m;
    int n;
    string activation_function;
    float * chain_deriv;//chain rule from next layer  
    float * linear_transformation; //Result of Denseoperation prior to activation function output 
    float * d_activation;

    float * input_vector;
    float y; //Ground truth of the input vector; if input_vector = X_train, then y = y_train data
    float * output_vector;
    float **weights;
    float *bias;

    float loss = 0.0;
    // int m;
    // int n;
    
    // float * linear_transformation; //Result of Denseoperation prior to activation function output 
    // float * chain_deriv;
    void initialize_weights(int m, int n, bool show_outputs)
    { //m rows, n cols`
        //Memory allocation onto heap to prevent stack overflow
        this->m = m; //Store the Densesizes for future use
        this->n = n;
        //Initialize weight matrix
        this->weights = new float * [m];
        for (int j = 0; j < m; j++)
        {
            this->weights[j] = new float[n];
            this->bias = new float;//Initialize bias vector; num entries of bias === num rows of weight matrix//Initialize bias vector; num entries of bias === num rows of weight matrix

        }
         
        


        //Seed our weights with random values

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                this->weights[i][j] = float(rand())/float((RAND_MAX)) * 1.0; //Store random values in weights
                // Comment this out when not in use to save runtime;
                if(show_outputs == true){
                    cout <<"Weights: " <<  this->weights[i][j] << endl;//Check to see what weights and

                }
            }
            this->bias[i] = float(rand())/float((RAND_MAX)) * 1.0;//Store random values in bias vector
            if(show_outputs == true){
                cout << "Bias: " << this->bias[i] << endl;
            }
        }
    }

    //Linear transformation of the input vector plus feeding the resulting vector components into an activation value
    void forward(float * input_vector, float y, bool show_output){
        
        int k = 0;
      
        // cout << "Input size: " << sizeof(input_vector)/sizeof(*input_vector) << endl;
        // cout << "Layer size: " << this->n << endl;
        this->input_vector = input_vector;
        this->output_vector = new float;
        this->linear_transformation = new float;
        float element;
        float activation_value;
        for(int i = 0; i < this->m; i++){
            element = 0.0;
            for(int j = 0; j < n; j++){
                element += this->weights[i][j] * input_vector[j];  
            }                  
            if(this->activation_function == "sigmoid"){
                // this->activation_function = "sigmoid";
                activation_value = sigmoid(element + this->bias[i]);
                this->loss = loss_i(y, activation_value);//Add to the loss function from output layer
            }else if(this->activation_function == "relu"){
                // this->activation_function = "relu";
                // activation_value = relu(element + this->bias[i]);
                activation_value = relu(element + this->bias[i]);
            }
            this->linear_transformation[i] = element + this->bias[i]; //Store the linear transformation result
            this->output_vector[i] = activation_value;   //Linear transform from matrix multiplication plus bias 
            if(show_output == true){
                cout << "Output vector elements: " << output_vector[i] << endl;
            }

        }             
        
    }




    void backpropagate(float learning_rate){
        //Take into account the final layer, which does not have a next node in the linked list
        if(this->next != NULL){//
            this->chain_deriv = matrix_outer_product(this->next->weights, this->next->chain_deriv, this->next->m, this->next->n);
            // Update d_activation to now deal with the derivatives of the previous activation
            this->d_activation = vector_derivative_of_activation(this->linear_transformation, this->m, relu);
        }
        else{//If next node is NULL, then that means we're backpropagating directly from the immediate derivative
            //of the scalar loss function... 
            float ground_truth_y = this->y; //Test value for now...
            //Define chain derivative if it's the final layer since there won't initially be a value for it 
            //If this layer is the output layer, then we need to calculate the full loss function at each epoch.


            this-> chain_deriv = vector_derivative_of_loss(this->output_vector, this->m, ground_truth_y, loss_i);
            this->d_activation = vector_derivative_of_activation(this->linear_transformation, this->m, sigmoid);
            // this->chain_deriv = einstein_sum(this->chain_deriv, this->d_activation, this->m);
            
        }
        // Update d_activation to now deal with the derivatives of the previous activation
        // this->d_activation = vector_derivative_of_activation(this->linear_transformation, this->m, relu);
        this->chain_deriv = einstein_sum(this->chain_deriv, this->d_activation, this->m);
        float **dW= tensor_product(this->chain_deriv, this->input_vector, this->m, this->n); //The nullptr to input_vector
        float *db = this->chain_deriv;
        learning_rate = -1 * learning_rate;
        dW = multiply_matrix_by_scalar(dW, learning_rate, this->m, this->n);
        db = multiply_vector_by_scalar(db, learning_rate, this->m);

        this->weights = add_matrices(this->weights, dW, this->m, this->n);
        this->bias = add_vectors(this->bias, db, this->m);

        // cout << "Change in weights of current layer... " << endl;
        // print_array(dW1, this->m, this->n);

    }
};




class Model{
    public:


        Dense * input_layer;
        Dense * output_layer;
        float total_loss; //Get this for each epoch from the forward pass

        void add_layer(int m, int n, string activation_function, bool show){
            Dense * new_layer = new Dense;
            Dense * current_layer = this->input_layer;//Start with head of linked list, i.e. input layer
            new_layer->m = m;
            new_layer->n = n;
            new_layer->activation_function = activation_function;
            new_layer->initialize_weights(m, n, false);
            //Case when no input layer exists and we're initializing the model for the first time
            // if(this->input_layer == NULL){
            //     this->input_layer = new_layer;
            // }
            
            while(current_layer->next != NULL){
                current_layer = current_layer->next;
            }//Exit loop to get last layer of model 

            new_layer->prev = current_layer; //Set a prev pointer so that with every added layer, we 
                                             //can backpropagate!
            current_layer->next = new_layer;

            this->output_layer = current_layer->next;
            
            if(show){
                cout << "Successfully added new layer of size: " << m << ", " << n << endl;

            }
        }

        //Forward pass of the network  

        void forward_pass(float * input_vector, float y, bool isTraining){
            //Traverse through the linked list and call the forward() function of each layer
            Dense * current = this->input_layer;
            float * current_vector = new float;
            float current_truth = y;//Truth label; in multilabel models this becomes one-hot encoding 
            current_vector = input_vector;
            // int counter = 0;
            while(current != NULL){
                // cout << current->m << ", " << current->n << endl;
                current->forward(current_vector, current_truth, false);

                if(isTraining == true){
                    this->total_loss += current->loss; //Update the loss with successive vectors added
                }
                
                current = current->next;
                // counter++;
            }
            // cout << counter << endl;
            // print_vector(current_vector, current->m);

            
        }

        void backpropagation(float learning_rate){
            Dense * current = this->output_layer;
            // cout << "The dimensions of the output layer in order are: " << current->m << ", " << current->n << endl;
            // cout << "The dimensions of the mid layer in order are: " << current->prev->m << ", " << current->prev->n << endl;

            while(current != NULL){
                current->backpropagate(learning_rate);
                // cout << "The dimensions of each layer in order are: " << current->m << ", " << current->n << endl;
                current = current->prev;
            }
        }

        void train(int epochs, int batch_size, float learning_rate, float *** training_data, int size_training_data){
            int EPOCHS = 10;
            //Lets train a batch of these at once!
            float * current_input_vector = new float;
            float current_y; 
            int size_of_dataset = size_training_data;
            int num_batches = int(size_of_dataset/batch_size);
            int start = 0;
            int end = 10;
            for(int e = 0; e < EPOCHS; e++){

                for(int index = start; index < end; index++){//counter is the size of our dataset; 
                                                            //in this case, counter is our batchsize
                    current_input_vector = training_data[index][0]; 
                    current_y = training_data[index][1][0]; //The index of data[1] is itself a 1 x 1 array so use [0] to get float
                    this->forward_pass(current_input_vector, current_y, true); //Run the forward pass!
                    this->backpropagation(0.001);
                }
         

            cout << "Completed Epoch " << e << endl;
            cout << "Training loss is: " << this->total_loss << endl;
            this->total_loss = 0.0; //Reset the loss function after each epoch otherwise you'll just be adding successive 
                                    //losses, and it won't be able to decrease over time.   
            end += batch_size;
            start += batch_size;

            }
        }

        float * predict(float * input_vector){
            float * y_prediction = new float;
            //Ground truth value is included but not needed
            this->forward_pass(input_vector, 0.0, false);
            y_prediction = this->output_layer->output_vector;
            return y_prediction;
        }




};


class Node{
  public:
    int data;
    Node * next;
    
};

class List{
    public:
        Node * head;
        void add_node(int val){
            Node * new_node = new Node;
            new_node->data = val;
            Node * current_node = this->head;
            while(current_node->next != NULL){
                current_node = current_node->next;
            }
            current_node->next = new_node;
            // cout << current_node->next->data << endl;
            
            cout << current_node->next->data << endl;
        }
};








