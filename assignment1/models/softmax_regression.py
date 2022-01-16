# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (optional ReLU activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        #    2) Compute the gradient of the loss with respect to the weights        #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################
        # can call functions now
        # what is the network? one layer softmax relu..weird name
        # one layer means only output layer.
        # weights are initialized.
        # No bias term included. - should it be included?
        # Follow andrew NG format
        x_z=np.dot(X,self.weights['W1'])
        x_a=self.ReLU(x_z)
        # Converting relu scores to probability
        # softmax test case use non probability values
        # X_s=self.sigmoid(X_a)
        x_pred=self.softmax(x_a)
        loss=self.cross_entropy_loss(x_pred, y)
        accuracy = self.compute_accuracy(x_pred, y)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight by chain rule                  #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        # there are two parts of the code
        # no auto grad
        # need to use whiteboard.. this can take time
        # y is one hot encoded vector
        yy=np.zeros((len(y), self.num_classes))
        for i,x in enumerate(y):
            yy[i][x]=1
        
        # softmax is dependent on the number of classes
        # Kronecker delta.
        # need to read more. cant proceed
        # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        # https://www.gormanalysis.com/blog/neural-networks-a-worked-example/
        # issue of summation
        # https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
        # relu - matrix multiplication is causing issues
        dL_pred_wrt_dx_r=x_pred-yy #(64, 10)
        dx_r_wrt_dz=self.ReLU_dev(x_z) #(64, 10)
        dz_wrt_dw=X #(64, 784)
        # need to update gradient dictionary as well
        # need to watch videos.. too much knowledge gap
        # Chain rule
        # dl/dw=dl/dx_pred | dx_pred/dx_r | dx_r/dz | dz/dw
        # I think reverse activation requires simple multiplication
        relu_gradient_pass=np.multiply(dL_pred_wrt_dx_r,dx_r_wrt_dz)
        gradients=(1/len(X))*np.dot(dz_wrt_dw.T,relu_gradient_pass)
        self.gradients['W1']=gradients
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy





        


