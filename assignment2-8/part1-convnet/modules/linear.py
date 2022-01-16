"""
Linear Module.  (c) 2021 Georgia Tech

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


class Linear:
    """
    A linear layer with weight W and bias b. Output is computed by y = Wx + b
    """

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.in_dim, self.out_dim)
        np.random.seed(1024)
        self.bias = np.zeros(self.out_dim)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass of linear layer
        :param x: input data, (N, d1, d2, ..., dn) where the product of d1, d2, ..., dn is equal to self.in_dim
        :return: The output computed by Wx+b. Save necessary variables in cache for backward
        """
        out = None
        #############################################################################
        # TODO: Implement the forward pass.                                         #
        #    HINT: You may want to flatten the input first                          #
        #############################################################################
        # why is called affine?
        # should be similar to assignment 1
        # dot product only
        # shape looks weird though
        # 4D array
        # N is number of rows?
        # self.in_dim is 120 = 4*5*6
        # what is w shape..if you can flatten this 3d array to a vector
        # out dim is 3. so dimensionality needs to reduced from 120 to 3
        # there is no self.w.. ok.. self.weight is there
        # there is bias as well.. what is the size of bias.. should be 3..yep
        # problem is np.dot of uneven sizes of matrix.should work.. assuming numpy developer thought of this
        # x is 2,4,5,6
        # w is 120,3
        # so it is expecting flattening - 4*120,120*3 
        # this should work ideally.. not flatten will be vector.. better to use reshape
        # actually, number of rows is 2.. not 4..
        # add bias as well
        out=np.dot(x.reshape(len(x),self.in_dim),self.weight)+self.bias
        # should work
        # lol..it worked..hopefully.. data will match
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        Computes the backward pass of linear layer
        :param dout: Upstream gradients, (N, self.out_dim)
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the linear backward pass.                                 #
        #############################################################################
        # now this will be funny
        # same logic as relu
        # nope.. this one requires updating gradient library.. okay
        # would that mean.. L2 regularization as well?
        # actually.. that is outside.. it is not part of loss
        # it is ..forgot the term.. weight decay..
        # so should be fine
        # dx, db and dw are none now..
        # this will not be np.multiply..
        # check shape of dout..
        # i am hungry.. need to have dinner
        # 10,5
        # self.weight.shape is 6,5
        # shape would be painfull..
        # generally.. it is old value*gradient..
        # so..it should be nope..
        # (N, self.out_dim) - so 10 rows of 5 dimensionality..10 should not change..dout should be first in np.dot
        # is there any activation?
        # why does it need dx? i dont remember this in assignment 1.
        # it was only weight and bias
        # should it be 0? may be a trick question.. put 0 and run the test case
        # (10, 2, 3)
        # this is the shape.. weight 
        # this will be w only..but.. that 6,5..so..shape does not match
        # think harder.. 10,2,3 is x shape..so dx_num is correct shape only
        # this should be solvable
        # multiply with dout as well dumbo
        # this random transposing is making me sick
        # irrespective of reshape, sum should be match
        # sum matches.. so..it looks like.. reshape is incorrect..
        # need to find the reason
        # dx=np.dot(self.weight,dout.T).T.reshape(x.shape)
        # failing for 100 rows

        # N,row,column=x.shape
        flattened_x=x.reshape(len(x),self.in_dim)
        flattened_dx=np.zeros([len(x),self.in_dim])
        
        flattened_dx=np.dot(dout,self.weight.T)
        
        self.dx=flattened_dx.reshape(x.shape)
        # dx=dout_flatten.dot(wout_flatten)

        # lets test it once.. this one was actually harder than the remaining two.. unexpected actually

        # flatten?
        # (6, 5) - self.weight
        # what is dout shape? 10,5
        # will have to take transpose to match 10
        dw=np.dot(x.reshape(len(x),self.in_dim).T,dout)
        # 6,5.. what was N.. N wont matter here.. this should match shape of self.weight
        db=np.sum(dout,axis=0)

        # self.dx=dx
        self.dw=dw
        # bias is (5,).. there are 5 models
        # changed reshape for db. 
        self.db=db
        # cant do reshape.. test case will fail
        # .reshape(self.weight.shape[1],1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
