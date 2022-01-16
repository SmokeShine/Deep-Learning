"""
MyModel model.  (c) 2021 Georgia Tech

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
from torch.nn.init import xavier_uniform_

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        # 
        # Keep it simple 
        # CNN blocks and end with linear ..
        # focus on part 1 error
        self.conv2d1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=0)
        self.conv2d2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=0)
        self.conv2d3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=0)
        self.relu=nn.ReLU()
        self.max_pool=nn.MaxPool2d(kernel_size=2,stride=2)
        # exponentially drop features
        # what is the size? This will take a lot of compute
        # sad
        self.linear1=nn.Linear(in_features=512,out_features=256)
        self.linear2=nn.Linear(in_features=256,out_features=128)
        self.linear3=nn.Linear(in_features=128,out_features=64)

        self.output=nn.Linear(in_features=64,out_features=10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        x1_cnn=self.conv2d1(x)
        x1_relu=self.relu(x1_cnn)
        x1_max_pool=self.max_pool(x1_relu)

        x2_cnn=self.conv2d2(x1_max_pool)
        x2_relu=self.relu(x2_cnn)
        x2_max_pool=self.max_pool(x2_relu)

        x3_cnn=self.conv2d3(x2_max_pool)
        x3_relu=self.relu(x3_cnn)
        x3_max_pool=self.max_pool(x3_relu)

        # import pdb;pdb.set_trace()
        x3_max_pool_reshaped=x3_max_pool.reshape(len(x3_max_pool),-1)
        # Spam linear layer till accuracy hits the benchmark
        # import pdb;pdb.set_trace()
        x4_linear1=self.linear1(x3_max_pool_reshaped)
        x4_linear2=self.linear2(x4_linear1)
        x4_linear3=self.linear3(x4_linear2)

        out=self.output(x4_linear3)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
