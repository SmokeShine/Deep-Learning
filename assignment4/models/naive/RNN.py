"""
Vanilla RNN Model.  (c) 2021 Georgia Tech

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

import numpy as np
import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """ An implementation of vanilla RNN using Pytorch Linear layers and activations.
        You will need to complete the class init function, forward function and hidden layer initialization.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
                output_size (int): the size of the output layer

            Returns: 
                None
        """
        super(VanillaRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #############################################################################
        # TODO:                                                                     #
        #    Initialize parameters and layers. You should                           #
        #    include a hidden unit, an output unit, a tanh function for the hidden  #
        #    unit, and a log softmax for the output unit.                           #
        #    You MUST NOT use Pytorch RNN layers(nn.RNN, nn.LSTM, etc).             #
        #############################################################################
        
        # this one is called first.. do i need to use torch text?
        # i dont even know what torch text does
        # lol.. torch text is not even imported
        # input size is 2; hidden size is 4; output size is 3;;
        # is this one hot encoded?it is init..
        # do whateven is written in comment;; this looks bad
        # hidden and output should be linear
        # do i need to add bias? try both?

        self.tanh=nn.Tanh()
        self.softmax=nn.LogSoftmax(dim=1)

        self.b2_linear=nn.Linear(input_size+hidden_size,hidden_size)
        self.b1_linear=nn.Linear(input_size+hidden_size,output_size)
        
        
        # log softmax for the output unit
        

        # import pdb;pdb.set_trace()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input, hidden):
        """ The forward function of the Vanilla RNN
            Args:
                input (tensor): a batch of data of shape (batch_size, input_size) at one time step
                hidden (tensor): the hidden value of previous time step of shape (batch_size, hidden_size)

            Returns:
                output (tensor): the output tensor of shape (batch_size, output_size)
                hidden (tensor): the hidden value of current time step of shape (batch_size, hidden_size)
        """

        output = None

        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass for the Vanilla RNN. Note that we are only   #
        #   going over one time step. Please refer to the structure in the notebook.#                                              #
        #############################################################################
        # import pdb;pdb.set_trace()
        # unfortunately, the call is from notebook and i cannot use the debugger here
        # can I?
        # it says line 56 is called..ok...so not this
        # this is a simple forward.. trust pytorch
        # there is input and hidden tensor.. that is fine..
        x_concat=torch.cat((input,hidden),dim=1)
        # check shape
        b1_linear=self.b1_linear(x_concat)
        output=self.softmax(b1_linear)

        b2_linear=self.b2_linear(x_concat)
        hidden=self.tanh(b2_linear)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
