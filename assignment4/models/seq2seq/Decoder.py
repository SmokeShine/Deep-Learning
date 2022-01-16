"""
S2S Decoder model.  (c) 2021 Georgia Tech

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

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        # i think decoder input should be of size 1..
        # this is confusing.. why do we have encoder hidden size here
        # when it was passed through multiple linear layers
        # if encoder and decoder sizes are same, they should be one variable
        # can the decoder size be more than 1?
        # why not?
        # i dont have input length here..
        # actually.. this should be max size of vocab
        # hardcoded to 10.. but this is incorrect
        
        
        self.embedding_layer=nn.Embedding(output_size,emb_size)
        # import pdb;pdb.set_trace()
        # self.embedding_layer.weight.requires_grad=True
        if self.model_type=="RNN":
            self.model=nn.RNN(input_size=emb_size,hidden_size=decoder_hidden_size,batch_first=True)
        elif self.model_type=="LSTM":
            # okay.. batch first non sense
            self.model=nn.LSTM(input_size=emb_size,hidden_size=decoder_hidden_size,batch_first=True)
            # we are not using linear on cell state
            # self.linear_ct=nn.Linear(decoder_hidden_size,output_size)
        
        self.linear=nn.Linear(decoder_hidden_size,output_size)
        self.softmax=nn.LogSoftmax(dim=2)
        self.dropout=nn.Dropout(dropout)
        
        
        
        # output size need to match with input of decoder
        # check the decoder function
        
        # softmax or log softmax?
        # to check dim.. remove log and do sum
        # should be across the last dimension which has the same size as output
        # i think deprecation uses the last dimension
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, 1); HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply linear layer and softmax activation to output tensor before   #
        #       returning it.                                                       #
        #############################################################################
        # import pdb;pdb.set_trace()
        # output, hidden = None, None
        # okay.. think properly.. we already have a local working encoder.. atleast local.. GS is another story
        # little tired?? wtf? most likely because of interview
        # and anxiety from job
        # calm... think something good
        # embedding code should be same
        # using embedding here does not make sense
        # why? because it is not int
        # there is no self.input_size
        # is there something in decoder arguments?
        # it looks same..but?> order of argument is different
        
        # import pdb;pdb.set_trace()
        # seq2seq - 1d is coming..
        
        # hidden_org=hidden.clone()
        # import pdb;pdb.set_trace()
        if input.ndim==1:
            input=input[:,None]
        # try:
        x_embed = self.embedding_layer(input)
        # except:
        #     import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        
        x_embed_drop=self.dropout(x_embed)
        # x_embed_drop=torch.tensor([[[-0.2849,  0.5639,  0.4996, -0.9857,  0.9159,  1.3820, -1.2060,
        #   -0.5790, -1.4635,  0.0492, -0.6932,  0.3769, -1.2960, -1.3341,
        #   -0.4206, -0.8895,  0.5765,  2.3705,  2.2101, -0.5096, -0.0836,
        #   -1.0805,  0.0000,  1.9224, -0.6579,  0.6970, -0.1786, -1.2044,
        #   -0.0000, -0.9101, -0.0000, -0.3578],
        #  [-0.2849,  0.5639,  0.0000, -0.9857,  0.9159,  0.0000, -1.2060,
        #   -0.5790, -0.0000,  0.0492, -0.6932,  0.3769, -1.2960, -1.3341,
        #   -0.4206, -0.8895,  0.0000,  2.3705,  2.2101, -0.5096, -0.0836,
        #   -1.0805,  0.0000,  0.0000, -0.6579,  0.6970, -0.0000, -1.2044,
        #   -1.9242, -0.9101, -0.0000, -0.0000]]])
        if self.model_type=="RNN":
            # import pdb;pdb.set_trace()
            
            x_output,hidden=self.model(x_embed_drop,hidden)
            
        else:
            # i think it is better to keep as tuple
            # even while returning
            # import pdb;pdb.set_trace()
            x_output,hidden=self.model(x_embed_drop,hidden)
            
            # for LSTM decoder,hidden and x_cell_State are not used

        # this is the failed test case.. good luck controlloing randomness
        # there are weights even inside RNN
        # x_output=torch.tensor([[[ 0.2195,  0.6580, -0.3123, -0.7027, -0.7242,  0.0258,  0.8812,
        #    0.0175,  0.5071,  0.7714,  0.3850,  0.1374,  0.3247, -0.7824,
        #    0.6561, -0.3684,  0.0835, -0.7656,  0.3064, -0.5660, -0.3512,
        #   -0.0443, -0.7157,  0.7148, -0.4731, -0.6524,  0.1019, -0.6684,
        #   -0.4126,  0.0792, -0.4949,  0.5324],
        #  [ 0.3431,  0.4995,  0.1057, -0.4831, -0.3937, -0.6439,  0.8976,
        #    0.3675,  0.7722,  0.9435,  0.3465, -0.0893, -0.3041, -0.8074,
        #    0.7007, -0.7437,  0.6749, -0.4717,  0.0894,  0.1581,  0.3113,
        #   -0.4596, -0.5905,  0.5638, -0.2138, -0.5267,  0.0717,  0.0268,
        #   -0.1772,  0.8157, -0.1484,  0.7428]]])
        # print(x_output)
        # no need to use hidden layer now
        
        Linear=self.linear(x_output)
        # Linear=torch.tensor([[[ 0.0794, -0.0198,  0.0135,  0.0864, -0.0906, -0.3845,  0.6702,
        #   -0.2117],
        #  [-0.2359, -0.2708,  0.2968,  0.2542, -0.1522, -0.0332,  0.7308,
        #   -0.5203]]])
        # print(Linear)
        # HINT: encoded does not mean from encoder!!
        # what does it mean? hmm.. i dont have context tbh
        # this is stupid.. encoded means from encoder only
        # i am hungry? i had tea.. so most likely thirsty
        # okay.. between encoder and decoder..
        # we can other blocks .. i think that will be in seq2seq
        # only the last time step? actually.. it will be last only
        # does not look like.
        # import pdb;pdb.set_trace()
        # lol .. lstm will create havoc here
        
        output=self.softmax(Linear)[:,0,:]
        
        # print(x_embed_drop,hidden_org,hidden,x_output,Linear,output)
        # print(x_embed_drop)
        # print(hidden_org)
        # print(x_output)
        # print(Linear)
        # print(output)
        # ttt=torch.tensor([[[-2.3975, -2.4323, -1.8648, -1.9074, -2.3138, -2.1948, -1.4308, -2.6819]]])
        # trying to get the test cases from gs
        # print(torch.seed())
        # what a shame.. i have to poison the stderr to find the sort order
        # beyond wtf
#         {'training': True, '_parameters': OrderedDict(), 
# '_buffers': OrderedDict(), '_non_persistent_buffers_set': set(), '_backward_hooks': OrderedDict(), '_is_full_backward_hook': None, '_forward_hooks': OrderedDict(), '_forward_pre_hooks': OrderedDict(), '_state_dict_hooks': OrderedDict(), '_load_state_dict_pre_hooks': OrderedDict(), '_modules': OrderedDict([('embedding_layer', Embedding(8, 32)), ('model', RNN(32, 32, batch_first=True)), ('linear', Linear(in_features=32, out_features=8, bias=True)), ('dropout', Dropout(p=0.2, inplace=False)), ('softmax', LogSoftmax(dim=2))]), 'emb_size': 32, 'encoder_hidden_size': 32, 'decoder_hidden_size': 32, 'output_size': 8, 'model_type': 'RNN'}
# 2072280583385595107
# Test Failed: False is not true : Your out: tensor([[-2.3975, -2.4323, -1.8648, -1.9074, -2.3138, -2.1948, -1.4308, -2.6819]],
#        grad_fn=<SliceBackward0>)
# Expected: tensor([[-2.0636, -2.1628, -2.1295, -2.0566, -2.2336, -2.5275, -1.4728, -2.3547]])
        # if torch.abs(ttt.sum()-output.sum())<1e-4:
        #     try :
        #         xya
        #     except:
        #         # print(source)
        #         # print(self.__dict__)
        #         # print("*****")
        #         # print(hidden_org,hidden_org.shape,input,input.shape,self.emb_size,self.encoder_hidden_size,self.decoder_hidden_size,
        #         # self.output_size,self.dropout)
        #         print(torch.seed(),x_embed_drop,hidden_org,hidden,x_output,Linear,output)
                # print("*****")
                # print(torch.seed())
        # output shape is incorrect? why? do i need to squeeze?
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
