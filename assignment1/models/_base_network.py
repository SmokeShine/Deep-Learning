# Do not use packages that are not in standard distribution of python
import numpy as np
class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):

        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        '''
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        '''
        prob = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Calculate softmax scores of input images                            #
        #############################################################################
        # this one required row/column wise operation..
        # can brute force. already have the answer from the test case
        # it need to target across num_classes dimension
        # e^x or e^-x?? forgot first one..
        # axis=0 is across column..
        # axis=1 is across rows..
        # which one is this?
        # actually i can simply take index of num_classes
        # why am I getting scores? Probably because it is the last layer
        # rank 1 matrix :/
        # need to reshape then
        # is score a np array? should be .. list would be suicidal
        # Hail andrew NG
        # prob=np.exp(scores)/np.exp(scores).sum(axis=1).reshape(scores.shape[0],1)
        # numerical stable verison
        # https://deepnotes.io/softmax-crossentropy
        exps=np.exp(scores-np.max(scores,axis=1).reshape(scores.shape[0],1))
        prob=exps/np.sum(exps,axis=1).reshape(scores.shape[0],1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return prob

    def cross_entropy_loss(self, x_pred, y):
        '''
        Compute Cross-Entropy Loss based on prediction of the network and labels
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        '''
        loss = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement Cross-Entropy Loss                                        #
        #############################################################################
        # -ylogy.. but is it asking for backward step? there is only one return
        # so simple ylogy should be fine
        # it is matrix.. how does it matter? just add  across the num_classes 
        # and average across N
        # eyes hurt.. already too late.. need to sleep.. good momentum..
        # can finish this
        # cant..symmetrical shape..
        N, num_classes=x_pred.shape
        # need to fix this.. dont do in one single shot..make intermediate variables
        # what is the denominator - 2m or m?
        # y is multi class - 1,2,0
        # convert to one hot encoding?
        # this will always be 10
        yy=np.zeros((len(y), x_pred.shape[1]))
        for i,x in enumerate(y):
            yy[i][x]=1
        # x_pred horizontal sum is 1
        # remove log 10
        # after one hot encoding, the second 1-y component is not needed
        # most likely one vs all? because of softmax?
        # no, even for binary, sum of y logy should work
        # yes.. orielly chapter confirms
        # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
        # looks like log loss and cross entropy loss can be mathematically different
        # actually cross entropy loss is more general form ..better to use that everywhere
        loss=-1/N*(np.sum(yy*np.log(x_pred)))
        # AssertionError: 0.7323996394789707 != 0.937803 within 5 places (0.20540336052102937 difference)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss

    def compute_accuracy(self, x_pred, y):
        '''
        Compute the accuracy of current batch
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        '''
        acc = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the accuracy function                                     #
        #############################################################################
        # worked in one shot..most probably wrong.. need to write more tests
        # IMPORTANT.. this can break
        acc=np.mean(np.argmax(x_pred,axis=1)==y)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return acc

    def sigmoid(self, X):
        '''
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, layer size)
        '''
        out = None
        #############################################################################
        # TODO: Comput the sigmoid activation on the input                          #
        #############################################################################
        # same trick as relu.. nope.. this one requires using brain to check row and column
        # actually no.. that would be for softmax.. this one can work with numpy
        # need to apply element wise function.. relu was simpler and required only gating
        # need to trust broadcasting
        # lol.. it worked
        out=1/(1+np.exp(-X))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def sigmoid_dev(self, x):
        '''
        The analytical derivative of sigmoid function at x
        :param x: Input data
        :return: The derivative of sigmoid function at x
        '''
        ds = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the derivative of Sigmoid function                        #
        #############################################################################
        # why are they not asking to derive this?
        # yep.. is it element wise or cross product?
        # gradient needs to be scalar..so it should be dot product
        # lets try brute force without np functions
        # fails.. hmm..
        # this numpy is really funny to read.. wont work..most probably
        # need to think now.
        # can you calculate by hand? it is a small matrix only..
        # ds is none type?
        # it is x.. need to call normal sigmoid function here
        # why make double calls.. cache the value of sigmoid
        # a is for activation.. not a bad variable name?
        # passed.. hmm.. some relief.. can go back and watch tom mitchell videos now
        a=self.sigmoid(x)
        ds=np.multiply(a,np.subtract(1,a))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return ds

    def ReLU(self, X):
        '''
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the ReLU activation is applied to the input (N, layer size)
        '''
        out = None
        #############################################################################
        # TODO: Comput the ReLU activation on the input                          #
        #############################################################################
        # X parameter is painful - some comedy with N and layer_size
        # What is N? - number of rows? 
        # 3 layers means 3 models
        # what is happening in the pytorch source code? most likely this max comparison has to element wise
        # it does not depend on other rows and columns.. element wise should be good enough
        # THis is a shameful pandas work. as long as it works
        # lol.. it passes..
        out=np.where(X>0,X,0)
        # Traceback (most recent call last):
        # File "<string>", line 1, in <module>
        # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def ReLU_dev(self,X):
        '''
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: gradient of ReLU given input X
        '''
        out = None
        #############################################################################
        # TODO: Comput the gradient of ReLU activation                              #
        #############################################################################
        # Now?? What is _dev?
        # I can see this for all activation functions.
        # what is in the doc string? it looks very similar to docstring of normal relu
        # anything in the requirement doc that TA provided
        # okay.. it is derivative.. 
        # really? 
        # Yes.. it works.
        out=np.where(X>0,1,0)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
