# Do not use packages that are not in standard distribution of python
import numpy as np
np.random.seed(1024)
from ._base_network import _baseNetwork

class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()


    def _weight_init(self):
        '''
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        '''

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        '''
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        '''
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        # sigmoid for intermediate activations
        # This one has separate bias as well
        
        x_z=np.dot(X,self.weights['W1'])+self.weights['b1'] #(64,128)
        x_a1=self.sigmoid(x_z) #(64,128)
        x_z2=np.dot(x_a1,self.weights['W2'])+self.weights['b2'] #(64,10)
        # x_a2=self.sigmoid(x_a2)
        x_a2=self.softmax(x_z2) #(64,10)
        loss=self.cross_entropy_loss(x_a2, y)
        accuracy = self.compute_accuracy(x_a2, y)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        # Does this has softmax back prop? Yes
        
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################
        # self.gradients.keys()
        # dict_keys(['W1', 'b1', 'W2', 'b2']
        # Unlike the first one, this one requires four gradients
        # need to use whiteboard
        # this will always be 10
        yy=np.zeros((len(y), 10))
        for i,x in enumerate(y):
            yy[i][x]=1
        dz2=x_a2-yy #(64,10)
        
        dw2=np.dot(x_a1.T,dz2) #(128,10)
        db2=dz2
        # element wise.. np.dot should match with this
        dz1=np.dot(dz2,self.weights['W2'].T)*self.sigmoid_dev(x_z)
        dw1=np.dot(X.T,dz1)
        db1=dz1

        # bias -  one for each output value.. there are 10.
        self.gradients['W2']=(1/len(X))*dw2
        self.gradients['b2']=(1/len(X))*np.sum(db2,axis=0)
        self.gradients['W1']=(1/len(X))*dw1
        self.gradients['b1']=(1/len(X))*np.sum(db1,axis=0)
        # no * in dot. * is element wise. dot is matrix multiplication
        # dl/dz1 .X.t

        # self.gradients['W1']=(1/len(X))*np.dot(self.weights['W2'].T,dL_pred_wrt_dx_a2)
        # self.gradients['b1']=(1/len(X))*np.sum(dL_pred_wrt_dx_a2*dx_a2_wrt_dx_a1*dx_a1_wrt_dx_z*dx_z_wrt_db1,axis=0)
        # dL_pred_wrt_dx_a2=x_pred-yy #(64,10)
        
        # dx_a2_wrt_dw2=x_a1 #(64, 128)
        # dx_a2_wrt_db2=1
        # dx_a2_wrt_dx_a1=self.weights['W2'] #(128, 10)
        # dx_a1_wrt_dx_z=self.sigmoid_dev(x_z) #(64, 128)
        # dx_z_wrt_dw1=X #(64, 784)
        # dx_z_wrt_db1=1 
        # # bias -  one for each output value.. there are 10.
        # self.gradients['W2']=(1/len(X))*np.dot(dx_a2_wrt_dw2.T,dL_pred_wrt_dx_a2)
        # self.gradients['b2']=(1/len(X))*np.sum(dL_pred_wrt_dx_a2*dx_a2_wrt_db2,axis=0)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


        return loss, accuracy


