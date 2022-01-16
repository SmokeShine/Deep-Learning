"""
CovNet Module.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

from .softmax_ce import SoftmaxCrossEntropy
from .relu import ReLU
from .max_pool import MaxPooling
from .convolution import Conv2D
from .linear import Linear

import numpy as np

class ConvNet:
    """
    Max Pooling of input
    """
    def __init__(self, modules, criterion):
        self.modules = []
        for m in modules:
            if m['type'] == 'Conv2D':
                self.modules.append(
                    Conv2D(m['in_channels'],
                           m['out_channels'],
                           m['kernel_size'],
                           m['stride'],
                           m['padding'])
                )
            elif m['type'] == 'ReLU':
                self.modules.append(
                    ReLU()
                )
            elif m['type'] == 'MaxPooling':
                self.modules.append(
                    MaxPooling(m['kernel_size'],
                               m['stride'])
                )
            elif m['type'] == 'Linear':
                self.modules.append(
                    Linear(m['in_dim'],
                           m['out_dim'])
                )
        if criterion['type'] == 'SoftmaxCrossEntropy':
            self.criterion = SoftmaxCrossEntropy()
        else:
            raise ValueError("Wrong Criterion Passed")

    def forward(self, x, y):
        """
        The forward pass of the model
        :param x: input data: (N, C, H, W)
        :param y: input label: (N, )
        :return:
          probs: the probabilities of all classes: (N, num_classes)
          loss: the cross entropy loss
        """
        probs = None
        loss = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement forward pass of the model                                 #
        #############################################################################
        # this one cannot be done directly by the test case. 
        # Finish simpler things first
        # Watch vectorization video. should be about reshare 4d to 2d to 4d back

        # i dont know what is the model expecting here
        # self.modules is a list
        # what is the rationale behind storing the model object in a list?
        
        # how many forward and backward passes? do i need to run in a loop
        # out=self.modules[0].forward(x)
        input_=x.copy()
        # print("===========FORWARD===========")
        for i in self.modules:
            # print(self.modules[i])
            # if input_.ndim>4:
            #     input_=np.squeeze(input_,0)
            # import pdb;pdb.set_trace()
            # try:
            #     print("W:",self.modules[i].weight.sum())                
            #     print("B:",self.modules[i].bias.sum())

            #     print("dW:",self.modules[i].dw.sum())
            #     print("dB:",self.modules[i].db.sum())
            #     print("dx:",self.modules[i].dx.sum())
            # except:
            #     try:
            #         print("dx:",self.modules[i].dx.sum())
            #     except:
            #         pass

            z=i.forward(input_)
            input_=z.copy()
        # will this give probability?
        # this are weights? there are negative values as well so cant be probability
        # The output computed by Wx+b
        # need to put sigmoid on top of it. what is the shape of y? is it binary classification?
        # it is softmax
        # SoftmaxCrossEntropy is provided
        # giving both
        self.softmax=SoftmaxCrossEntropy()
        probs, loss=self.softmax.forward(input_,y)
        # where is the function?
        # y needs to be one hot encoded.
        # i know it will be 10
        # yy=np.zeros((len(y), 10))
        # for i,x in enumerate(y):
        #     yy[i][x]=1
        # how will you calculate loss function?
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return probs, loss

    def backward(self):
        """
        The backward pass of the model
        :return: nothing but dx, dw, and db of all modules are updated
        """
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement backward pass of the model                                #
        #############################################################################
        # sotfmax would require passing gradients back
        # import pdb;pdb.set_trace()
        # print("===========BACKWARD===========")
        self.softmax.backward()
        # how will it update dx,dw and db?
        # print("Softmax dx:",self.softmax.dx.sum())
        dout=self.softmax.dx.copy()
        
        reversed_=self.modules[::-1]
        for i in reversed_:
            # print(i)
            # backward doesnot return anything
            i.backward(dout)
            # why dx dw or db?
            # import pdb;pdb.set_trace()
            # only dx part becomes dout for the last layer.. need to understand more.
            # most likely because of chain rule
            dout=i.dx.copy()
            # try:
            #     print("W:",i.weight.sum())
            #     print("B:",i.bias.sum())
            #     print("dW:",i.dw.sum())
            #     print("dB:",i.db.sum())
            #     print("dx:",i.dx.sum())
            # except:
            #     try:
            #         print("dx:",i.dx.sum())
            #     except:
            #         pass
            # import pdb;pdb.set_trace()
            # maxpool will not have any parameters to backpropagate
            # dw=self.modules[len(self.modules)-i-1].dw
            # db=self.modules[len(self.modules)-i-1].db
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################