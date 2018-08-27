from builtins import range
from builtins import object
import numpy as np

from ecbm4040.layer_funcs import *
from ecbm4040.layer_utils import *

class MLP(object):
    """
    MLP with an arbitrary number of dense hidden layers,
    and a softmax loss function. For a network with L layers,
    the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-2):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        
        self.layers = layers

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        num_layers = self.num_layers
        layers = self.layers
        ###################################################
        #TODO: Feedforward                                #
        ###################################################
        output = []
        output.append(layers[0].feedforward(X))
        for i in range(1, num_layers):
            output.append(layers[i].feedforward(output[i-1]))
            
        output[num_layers - 1] = output[num_layers - 1] - np.max(output[num_layers - 1])
        step3 = softmax_loss(output[num_layers - 1], y)
        loss = step3[0]

        
        ###################################################
        #TODO: Backpropogation                            #
        ###################################################
        
        dX = step3[1]
        back = []
        back.append(layers[num_layers-1].backward(dX))
        
        for i in range(1, num_layers):
            back.append(layers[num_layers-1-i].backward(back[i-1]))
        
        #dense_back = layer1.backward(affine_back)
        self.layers = layers
        
        
        ###################################################
        # TODO: Add L2 regularization                     #
        ###################################################
        #square_weights = np.sum(layer1.params[0]**2) + np.sum(layer2.params[0]**2)
        square_weights = 0
        for i in range(num_layers):
            square_weights = square_weights + np.sum(layers[i].params[0]**2)
        loss += 0.5*self.reg*square_weights
        return loss
        
        
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
        
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        ###################################################
        #TODO: Use SGD or SGD with momentum to update     #
        #variables in layers                              #
        ###################################################
        # creates new lists with all parameters and gradients
        #layer1, layer2 = self.layer1, self.layer2
        layers = self.layers
        num_layers = self.num_layers
        params = []
        grads = []
        for i in range(num_layers):
            params += layers[i].params
            grads += layers[i].gradients
            
        
        # Add L2 regularization
        reg = self.reg
        grads = [grad + reg*params[i] for i, grad in enumerate(grads)]
        
        for i in range(2*num_layers):
            params[i] = params[i] - learning_rate*grads[i]
        
        


        
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
   
        # update parameters in layers
        for i in range(num_layers):
            self.layers[i].update_layer(params[2*i:2*(i+1)])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers
        #######################################################
        #TODO: Remember to use functions in class SoftmaxLayer#
        #######################################################
        output = []
        output.append(layers[0].feedforward(X))
        for i in range(1, num_layers):
            output.append(layers[i].feedforward(output[i-1]))
            
        output[num_layers - 1] = output[num_layers - 1] - np.max(output[num_layers - 1])
                
        
        y_pred = np.argmax(output[num_layers - 1], axis = 1)
        predictions = y_pred


        
        #######################################################
        #                 END OF YOUR CODE                    #
        #######################################################
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc
        
        


