"""
Vanilla CNN model.  (c) 2021 Georgia Tech

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


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        # passed two layer. If I can fix this quickly, I can try to modify this to
        # fix part 1
        # no input dim?
        # may number of filters calculation is required to be done manually
        # hard coding in_channels as 3
        self.conv2d=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=7,stride=1,padding=0)
        self.relu=nn.ReLU()
        self.max_pool=nn.MaxPool2d(kernel_size=2,stride=2)
        # what is the size of hidden feature here?
        hidden_size=5408
        self.output=nn.Linear(in_features=hidden_size,out_features=10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        
        # x=x.reshape(len(x),-1)
        x=self.conv2d(x)
        x=self.relu(x)
        x=self.max_pool(x)
        # import pdb;pdb.set_trace()
        # torch.Size([128, 32, 13, 13])
        # need to rename for boiler plate consistency
        # need to flatten before conenctingq
        x=x.reshape(len(x),-1)
        # import pdb;pdb.set_trace()
        outs=self.output(x)
        # weird.. no linear? this can n channel
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs
