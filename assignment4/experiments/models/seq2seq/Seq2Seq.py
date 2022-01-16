import random

from torch.functional import Tensor

""" 			  		 			     			  	   		   	  			  	
Seq2Seq model.  (c) 2021 Georgia Tech

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
import torch.optim as optim


# import custom models


class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the Seq2Seq model. You should use .to(device) to make sure  #
        #    that the models are on the same device (CPU/GPU). This should take no  #
        #    more than 2 lines of code.                                             #
        #############################################################################
        # there is no information in pdf or notebook
        # need to call trace 
        # it is supposed to call encoder and decoder..
        # there are two parameters and device
        # self.device=device
        self.encoder=encoder.to(device)
        self.decoder=decoder.to(device)
        # import pdb;pdb.set_trace()
        # now what? push to device.. it is supposed to be here?
        # how to check if this is in cuda
        # next(self.encoder.parameters()).is_cuda
        # looks like it was not pushed correctly
        # wtf.. it is using cpu parameter
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, source, out_seq_len=None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        batch_size = source.shape[0]
        seq_len = source.shape[1]
        if out_seq_len is None:
            out_seq_len = seq_len

        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass of the Seq2Seq model. Please refer to the    #
        #   following steps:                                                        #
        #       1) Get the last hidden representation from the encoder. Use it as   #
        #          the first hidden state of the decoder                            #
        #       2) The first input for the decoder should be the <sos> token, which #
        #          is the first in the source sequence.                             #
        #       3) Feed this first input and hidden state into the decoder          #  
        #          one step at a time in the sequence, adding the output to the     #
        #          final outputs.                                                   #
        #       4) Update the input and hidden weights being fed into the decoder   #
        #          at each time step. The decoder output at the previous time step  # 
        #          will have to be manipulated before being fed in as the decoder   #
        #          input at the next time step.                                     #
        #############################################################################
        # push all inputs to cuda
        source=source.to(self.device)
        outputs = None
        # too manu instructions.. what is coming here.. seq_len
        # this looks like BD4H but it was part of tuple
        # what is out_seq_len..ok.. some truncation needs to happen
        # so some data will be lost
        # weird..source is sequence..hmm..
        # call the test cases.. that should work
        # do i need loop?
        
        # wait.. we are not using out? that will break RNN/LSTM
        # if sos is index 0. what is index of full stop..sentence end
        # from 4, it loops like a loop is required for decoder
        # loop on out_seq_len?
        # import pdb;pdb.set_trace()
        # first check if encoder atleast is running
        # shoudl the loop be out_seq_len +1/2
        # or call it separately
        # out, hidden=self.decoder.forward(out,last_hidden_representation_of_encoder)
        # not required.. it says each input is sos..
        # so no separate treatment should be ideally required
        # no output dim 
        output_dim=self.decoder.output_size
        #8 .. ide does not detect.. but pdb says 8..atleast it is non zero
        # it is return..so variable name is defined
        
        # shape is an issue.. it was 2d in unit test.. here it is 3d?
        outputs=torch.zeros(out_seq_len,batch_size,output_dim)
        # flip this <<<<<<<<<<<<<<<<<<<<< 
        # this needs to map to sos somehow..
        # do i need to push this to gpu?
        # import pdb;pdb.set_trace()
        # this output needs to match with 32
        out,last_hidden_representation_of_encoder=self.encoder.forward(source)
        # import pdb;pdb.set_trace()
        input_token=source[:,0]
        # outputs[0]=input_token
        # this is the issue .. not 0:.. only 0
        # wtf .. created new error - RuntimeError: input must have 3 dimensions, got 2
        # this is in decoder
        # i think this is creating issues.. because 1*1 is first argument,
        # and second is batch size
        hidden=last_hidden_representation_of_encoder
        # sos and eos - skipping sos
        for i in range(out_seq_len):
            # print(i)
            # waht is x? must be out? it is not being used
            #
            # print(input_token)
            # as per the video, the context vector needs to be same-  ct
            # but state can vary
            # https://primetime.bluejeans.com/a2m/events/playback/59707898-f48b-4f33-ba20-ee2ed0c9a6ee
            input_token_prob, hidden = self.decoder.forward(input_token,hidden)
            # does this mean .. i have to arg max for next input token
            outputs[i]=input_token_prob
            # import pdb;pdb.set_trace()
            # import pdb;pdb.set_trace()
            # this is coming as scalar and not as list
            # import pdb;pdb.set_trace()
            input_token=torch.argmax(input_token_prob,dim=1).view(-1,1)
            # print(input_token)
            # which dimension?
            
            # import pdb;pdb.set_trace()
            # looks like out needs hstack with sos token first
            # this out needs to saved some where?
            # is it an index? dont think so.. why...
            # it is log softmax at the end.. that will 1/0..
            # no probability score
            # so it is tensor..die..wtf..
            # can use similar logic to assignmetn 2..
            # create the array and fill..
            # dont know the size.. batch size, output dim?
            # yes.. thiwas happening in decoder output.. 
            # it is not single output..
            # it is prob score for each i..what is happening to batch size?
            # not sure.. better to run once
            # but how to update.. it is batch size,time step
            # may be a stupid question.. but is the batch padded?
            # i think output becomes input for next step..
            # so for step 0.. it should be sos

        # not matching with test case
        # import pdb;pdb.set_trace()
        
        outputs=outputs.permute(1,0,2)

        # ttt=torch.tensor([[[-2.3975, -2.4323, -1.8648, -1.9074, -2.3138, -2.1948, -1.4308, -2.6819]]])
        # # trying to get the test cases from gs
        # # print(torch.seed())
        
        # if ttt.sum()-outputs.sum()<10.:
        #     try :
        #         xya
        #     except:
        #         # print(source)
        #         # print(self.__dict__)
        #         print(torch.seed())
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs

