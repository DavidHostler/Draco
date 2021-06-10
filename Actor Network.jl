using Statistics 
using Printf
using Random 


#Implementing weights, biases and differentiation right away!

#Just initialize the weights and biases immediately 
#Input size 4, output 256, i.e. 4 cols 256 rowsq
W1 = rand(Float64, (256,4)) * 0.0000003
b1 = rand(Float64, (256))
println("First layer initialized!")

#256 * 256 hidden layer, and bias vector!
W2 = rand(Float64, (256,256)) * 0.0000003
b2 = rand(Float64, (256))
println("Second layer initialized!")

#256 inputs, 1 output per layer
W_x = rand(Float64, (1, 256))
W_y = rand(Float64, (1, 256))

b_x = rand(Float64, (1))
b_y = rand(Float64, (1))
print("Action layers initialized!")

#Activation functions
function ReLU(vector)
    x = Vector{BigFloat} 
    x = zeros(0)
    for element in vector
        if element >= 0
            append!(x, element)
        else
            append!(x, 0)
        end
    end
    return x
end


function ReLU_deriv(vector)
    x = Vector{BigFloat} 
    x = zeros(0)
    for element in vector
        if element >= 0
            append!(x, 1)
        else
            append!(x, 0)
        end
    end
    return x
end


function Sigmoid(x)  
         return 1/(BigFloat(1) + exp(-x)) 
    
end

#Compute Loss, Backpropagation and gradient descent

#Crossentropy loss
function log_loss(x)
    y = Vector{BigFloat} 
    y = zeros(0)
    for element in x
        append!(y, -log(element))
    end
    return y
end
#Derivative of lossd
function log_loss_deriv(x)  
        return -1/BigFloat(x)
    end  



function hadamard(A, B)
    o = Vector{Int64}
    o = zeros(0)  
    for i in 1:size(A)[1]
        append!(o, A[i] * B[i])
    end
    return o
end
        
#Generate set of inputs size 4 vector
inputs = Vector{BigFloat}
inputs = zeros(0)
for i in 1:4
    append!(inputs, i * 0.1)
end 


#Forward pass of the model

function Forward(inputs, W1, b1, W2, b2, W_x, b_x, W_y, b_y)
    Z1 = W1 * inputs + b1
    first_layer = ReLU(Z1)
    Z2 = W2 * first_layer + b2
    second_layer = ReLU(Z2)
    Z_x = W_x * second_layer + b_x
    scalar_input_x = Z_x
    Z_y = W_y * second_layer + b_y
    Accel_x = Sigmoid(Z_x[1])
    Accel_y = Sigmoid(Z_y[1])
end
 
 
#Compute Loss
Loss = log_loss(Accel_x)[1] +  log_loss(Accel_y)[1]
#Backpropagation
grad_loss_x = [log_loss_deriv(Accel_x)]
grad_loss_y = [log_loss_deriv(Accel_y)]
#Backpropagation
function Backpropagation(grad_loss_x, grad_loss_y, first_layer, second_layer, inputs)
    #dZ_x = dLoss / dZ_x = grad_loss
    dZ_x = grad_loss_x
    dZ_y = grad_loss_y

    #dW_x =  dZ_x * ReLU_deriv(prev_layer_Z2)
    dW_x =  zeros( 1,256 ) 
    dW_x +=   dZ_x * second_layer'

    dW_y = zeros(1, 256)
    dW_y = dZ_y * second_layer'

    dZ2 = hadamard(transpose(W_x) * dZ_x, ReLU_deriv(second_layer)) + hadamard(transpose(W_y) * dZ_y, ReLU_deriv(second_layer))  
    # had!(Dis with,  ReLU_deriv(first_layer))
    dW2 = zeros(256,256)
    dW2 +=   dZ2 * first_layer'

    dZ1 = hadamard(transpose(W2) * dZ2 , ReLU_deriv(first_layer))

    dW1 = dZ1 * inputs'
end


#Update the parameters
function update(alpha , Z_x, Z_y, W_x, W_y, Z2, W1, Z1,  
                        dZ_x, dZ_y, dW_x, dW_y, dW2, dZ2, dW1, dZ1)
    Z_x = Z_x - dZ_x * alpha
    Z_y = Z_y - dZ_y * alpha
    W_x = W_x - dW_x * alpha
    W_y = W_y - dW_y * alpha    
    Z2 = Z2 - dZ2 * alpha
    W1 = W1 - dW1 * alpha
    Z1 = Z1 - dZ1 * alpha
    W1 = W1 - dW1 * alpha
end    
    
 

#Gradient descent

alpha = 0.001

for i in 1:5
    Forward(inputs, W1, b1, W2, b2, W_x, b_x, W_y, b_y) 
    Backpropagation(grad_loss_x, grad_loss_y, first_layer, second_layer, inputs) 
    Z_x = Z_x - dZ_x * alpha
    Z_y = Z_y - dZ_y * alpha
    W_x = W_x - dW_x * alpha
    W_y = W_y - dW_y * alpha    
    Z2 = Z2 - dZ2 * alpha
    W1 = W1 - dW1 * alpha
    Z1 = Z1 - dZ1 * alpha
    W1 = W1 - dW1 * alpha
end
 