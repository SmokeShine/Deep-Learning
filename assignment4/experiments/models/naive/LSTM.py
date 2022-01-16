"""
LSTM model.  (c) 2021 Georgia Tech

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


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization

    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes as you wish here.                      #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   You also need to include correct activation functions                      #
        ################################################################################
        # ok... i need to run from notebook again..
        # import pdb;pdb.set_trace()
        # it is better to blindly follow the pdf
        # the worspace feature is cool tbh
        # okay.. i need to initialize the four gates
        # the forward will use the last two equations
        # weight matrix? there is init_hidden function..but it is called later
        # okay...so? i think i will have to create a torch tensor
        # why not linear?
        # i_t: input gate
        # bias is true by default
        
        # i think all the weight matrix of gate should be same
        # okay.. what is the shape of xt?
        # xt
        # [1,2][2,4]=[1,4]
        # ht
        # [1,4][4,4]=[1,4]

        # self.w_ii=nn.Linear(input_size,hidden_size,bias=False)
        # self.w_hi=nn.Linear(hidden_size,hidden_size,bias=False)

        # self.w_if=nn.Linear(input_size,hidden_size,bias=False)
        # self.w_hf=nn.Linear(hidden_size,hidden_size,bias=False)

        # self.w_ig=nn.Linear(input_size,hidden_size,bias=False)
        # self.w_hg=nn.Linear(hidden_size,hidden_size,bias=False)

        # self.w_io=nn.Linear(input_size,hidden_size,bias=False)
        # self.w_ho=nn.Linear(hidden_size,hidden_size,bias=False)

        ###########
        self.w_ii=nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.w_hi=nn.Parameter(torch.Tensor(hidden_size,hidden_size))

        self.w_if=nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.w_hf=nn.Parameter(torch.Tensor(hidden_size,hidden_size))

        self.w_ig=nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.w_hg=nn.Parameter(torch.Tensor(hidden_size,hidden_size))

        self.w_io=nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.w_ho=nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        
        self.b_ii=nn.Parameter(torch.ones(hidden_size))
        self.b_hi=nn.Parameter(torch.ones(hidden_size))

        self.b_if=nn.Parameter(torch.ones(hidden_size))
        self.b_hf=nn.Parameter(torch.ones(hidden_size))
        self.b_ig=nn.Parameter(torch.ones(hidden_size))
        self.b_hg=nn.Parameter(torch.ones(hidden_size))
        self.b_io=nn.Parameter(torch.ones(hidden_size))
        self.b_ho=nn.Parameter(torch.ones(hidden_size))

        # f_t: the forget gate

        # g_t: the cell gate

        # o_t: the output gate

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        h_t, c_t = None, None
        N,T,_=x.shape
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        # import pdb;pdb.set_trace()
        # line 68 is called.. initialization.. okay.. that is expected

        h_t=torch.zeros(N,self.hidden_size)
        c_t=torch.zeros(N,self.hidden_size)
        for t in range(T):
            
            # i_t=self.sigmoid(self.w_ii(x[:,t,:])+self.b_ii+self.w_hi(h_t)+self.b_hi)
            # f_t=self.sigmoid(self.w_if(x[:,t,:])+self.b_if+self.w_hf(h_t)+self.b_hf)
            # g_t=self.sigmoid(self.w_ig(x[:,t,:])+self.b_ig+self.w_hg(h_t)+self.b_hg)
            # o_t=self.tanh(self.w_io(x[:,t,:])+self.b_io+self.w_ho(h_t)+self.b_ho)

            # import pdb;pdb.set_trace()
            i_t=self.sigmoid(x[:,t,:]@self.w_ii\
                +self.b_ii\
                +h_t@self.w_hi\
                +self.b_hi)
            # import pdb;pdb.set_trace()
            f_t=self.sigmoid(x[:,t,:]@self.w_if\
                +self.b_if\
                +h_t@self.w_hf\
                +self.b_hf)
            # import pdb;pdb.set_trace()
            g_t=self.tanh(x[:,t,:]@self.w_ig\
                +self.b_ig\
                +h_t@self.w_hg\
                +self.b_hg)
            # import pdb;pdb.set_trace()
            o_t=self.sigmoid(x[:,t,:]@self.w_io\
                +self.b_io\
                +h_t@self.w_ho\
                +self.b_ho)
            # import pdb;pdb.set_trace()
            c_t=torch.mul(f_t,c_t)+torch.mul(i_t,g_t)
            h_t=torch.mul(o_t,self.tanh(c_t))
            # import pdb;pdb.set_trace()

        # x is 4,3,2.. batch of 4..3*2 matrix? which is the time component?
        # most likely 3 ok.. -1,-1, then 1,1, then 3,3

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
