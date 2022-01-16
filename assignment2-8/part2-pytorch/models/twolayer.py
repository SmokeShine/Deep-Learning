"""
Two Layer Network Model.  (c) 2021 Georgia Tech

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

import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        # this is similar to BD4H assignment..
        # looks like DL team gave code to BD4H. average meter is same
        # import pdb;pdb.set_trace()
        self.first_hidden=nn.Linear(in_features=input_dim,out_features=hidden_size)
        self.sigmoid=nn.Sigmoid()
        self.FullyConnectedOutput=nn.Linear(in_features=hidden_size,out_features=num_classes)
        # anything about initialization?
        # test case is not running the training loop
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        # what is x shape?
        # have to do it here as train is common for linear and CNN
        x=x.reshape(len(x),-1)
        x=self.first_hidden(x)
        x=self.sigmoid(x)
        # need to rename for boiler plate consistency
        out=self.FullyConnectedOutput(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
