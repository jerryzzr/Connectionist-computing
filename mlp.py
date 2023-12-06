import numpy as np

class MLP:
    # Initiallize attributes
    def __init__(self, NI, NH, NO):
        # number of inputs
        self.NI = NI
        # number of hidden units
        self.NH = NH
        # number of outputs
        self.NO = NO

        # array containing the weights in the lower layer
        self.W1 = np.random.randn(self.NI, self.NH)
        # array containing the weights in the upper layer
        self.W2 = np.random.randn(self.NH, self.NO)
        
        # arrays containing the weight *changes* to be applied onto W1 and W2
        self.dW1 = np.zeros((self.NI, self.NH))
        self.dW2 = np.zeros((self.NH, self.NO))
        # array containing the activations for the lower and upper layer – will need to keep track of these for when you have to compute deltas
        self.Z1 = np.zeros(self.NH)
        self.Z2 = np.zeros(self.NO)
        # array where the values of the hidden neurons are stored – need these saved to compute dW2
        self.H = np.zeros(self.NH)
        # array where the outputs are stored
        self.O = np.zeros(self.NO)
    
    # random the 2 layers weight into a small value and also set the changes in two weight to 0
    def randomise(self):
        self.W1 = np.random.randn(self.NI, self.NH)
        self.W2 = np.random.randn(self.NH, self.NO)
        self.dW1 = np.zeros((self.NI, self.NH))
        self.dW2 = np.zeros((self.NH, self.NO))

    # Performs the forward pass through the network, and can be sigmoidal or linear
    def forward(self, I):
        self.Z1 = np.dot(I, self.W1)
        self.H = 1 / (1 + np.exp(-self.Z1))
        self.Z2 = np.dot(self.H, self.W2)
        self.O = 1 / (1 + np.exp(-self.Z2))
        return self.O

    # Performs the backward pass to compute errors and weight updates
    def backward(self, I, targets):
        error = np.mean(np.square(targets - self.O))
        
        delta2 = (self.O - targets) * (self.O * (1 - self.O))
        # For XOR
        #self.dW2 += np.dot(self.H.T, delta2)
        # For Sin and special test
        self.dW2 += np.dot(self.H.reshape(-1, 1), delta2.reshape(1, -1))

        delta1 = np.dot(delta2, self.W2.T) * (self.H * (1 - self.H))
        # For XOR
        #self.dW1 += np.dot(I.T, delta1)
        # For Sin and special test
        self.dW1 += np.dot(I.reshape(-1, 1), delta1.reshape(1, -1))    
        return error
    
    # Updates the weights using the computed gradients and a specified learning rate
    def update_weights(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.W2 -= learning_rate * self.dW2
        self.dW1 = np.zeros_like(self.W1)
        self.dW2 = np.zeros_like(self.W2)

