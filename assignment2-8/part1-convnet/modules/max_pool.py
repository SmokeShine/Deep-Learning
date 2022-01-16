"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        # kernel and stride are property of maxpool
        # what is the image size? what is N.. what is C?
        # does not matter.. N,C will remain the same.. only HW should shrink
        # one way would be 
        # did you check vectorization office hour? this will require triple loop
        # channels.. rows and columns
        # N, C, H, W=x.shape
        # for n in N:
        #     for c in C:
        #         for 
        # make it simple.. N should remain the same.. C should remain the same
        # only H and W would require max operation
        # stride means for loop
        # what is the problem.. the problem is patch..it needs to go horizontally and vertically
        # and not sequential
        # no shame - do a double for loop and close this
        temp=[]
        H_out=[]
        W_out=[]
        row_movement=0
        
        N, C, H, W=x.shape
        for n in range(N):
            for c in range(C):
                channel_temp_array=x[n][c]
                row_,column_=channel_temp_array.shape
                # divisble by 2?
                row_movement=0
                for i in range(0,row_,self.stride):
                    row_movement+=1
                    column_movement=0
                    for j in range(0,column_,self.stride):
                        column_movement+=1
                        patch=channel_temp_array[i:i+self.stride,j:j+self.stride]
                        temp.append(np.max(patch))
                        position_max=np.unravel_index(patch.argmax(),patch.shape)
                        H_out.append(position_max[0])
                        W_out.append(position_max[1])
        
        temp_array=np.array(temp)
        max_pool_size_formula=((row_-self.kernel_size)//self.stride)+1
        out=temp_array.reshape(N,C,max_pool_size_formula,max_pool_size_formula)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        # what is dout shape
        # 3244 
        # weird shape.. it is upstream derviatives
        # only the one which provided the max value should have derivative as 1
        # rest in the patch should be zero
        # do this..loop through the image.. for a given patch, and max location, multiply with dout else
        # leave it as it is
        # can use the looping logic
        N, C, H, W=x.shape
        non_zero_patches=[]
        for n in range(N):
            for c in range(C):
                channel_temp_array=x[n][c]
                dout_temp_array=list(dout[n][c].flatten())
                row_,column_=channel_temp_array.shape
                for i in range(0,row_,self.stride):
                    column_movement=0
                    for j in range(0,column_,self.stride):
                        patch=channel_temp_array[i:i+self.stride,j:j+self.stride]
                        # ok..in this patch.. we need to use the max location
                        # how to know if it flatten correctly
                        # why not simply pop from the top? pop 0?
                        # Hout is an array or list?
                        max_i_in_patch=H_out[0]
                        H_out.pop(0)
                        max_j_in_patch=W_out[0]
                        W_out.pop(0)
                        # create a 0,0 array
                        _zero=np.zeros(patch.shape)
                        _zero[max_i_in_patch,max_j_in_patch]=1.
                        
                        popped=dout_temp_array[0]
                        non_zero=np.multiply(_zero,popped)
                        dout_temp_array.pop(0)
                        # now how to stitch back; just append the array now.. or we can simply overwrite the existing patch
                        # keep it different
                        non_zero_patches.append(non_zero)
                        # this is a masked image
        # what to return? 
        # what is in the test case? is it checking the original 
        # first recreate the image
        # non_zero_patchesq
        dx=np.zeros_like(x)
        for n in range(N):
            for c in range(C):
                for i in range(0,row_,self.stride):
                    for j in range(0,column_,self.stride):
                        dx[n,c,i:i+self.stride,j:j+self.stride]=non_zero_patches[0]
                        non_zero_patches.pop(0)
                                     
        self.dx=dx
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
