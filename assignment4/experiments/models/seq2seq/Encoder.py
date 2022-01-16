"""
S2S Encoder model.  (c) 2021 Georgia Tech

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


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN" and "LSTM".                                                #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden weights of the Encoder(namely, Linear - ReLU - Linear).   #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        # notebook again bro..
        # first it should be initialization.. then here
        # hmm.. looks like rnn and lstm can both be passes.. will that be a problem?
        # input remains the same..but..we do not initialize internals..
        # we can directly use pytorch functions..so?
        # start with simple
        # embedding layer - initialization for embedding layer? should it be non trainable?
        # this should be vocab size
        
        
        self.embedding_layer=nn.Embedding(input_size,emb_size)
        # i am not sure about this
        # it is not pretrained.. so should be true
        # self.embedding_layer.weight.requires_grad=True
        # self.model_type this is a string - put an if statement?
        if self.model_type=="RNN":
            self.model=nn.RNN(input_size=emb_size,hidden_size=encoder_hidden_size,batch_first=True)
        elif self.model_type=="LSTM":
            # okay.. batch first non sense
            self.model=nn.LSTM(input_size=emb_size,hidden_size=encoder_hidden_size,batch_first=True)
            # self.linear_ct=nn.Linear(encoder_hidden_size,encoder_hidden_size)
        
        
        self.linear=nn.Linear(encoder_hidden_size,encoder_hidden_size)
        self.relu=nn.ReLU()
        # don't know.. but this should be
        self.out=nn.Linear(encoder_hidden_size,decoder_hidden_size)
        # output size need to match with input of decoder
        # check the decoder function
        
        # what is the output size? batch length? 1? do we have batch length as argument?
        # is this predecided or need to be calculated here? how will we know the size of decoder
        # it has to be argument
        # do we need to use decoder hidden size here?
        # it does not make sense.. decoder is initialized later
        self.dropout=nn.Dropout(dropout)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder; later fed into the Decoder.
                hidden (tensor): the weights coming out of the last hidden unit
        """

        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #############################################################################
        self.tanh=nn.Tanh()
        output, hidden = None, None
        
        # https://colab.research.google.com/github/bala-codes/Natural-Language-Processing-NLP/blob/master/Neural%20Machine%20Translation/1.%20Seq2Seq%20%5BEnc%20%2B%20Dec%5D%20Model%20for%20Neural%20Machine%20Translation%20%28Without%20Attention%20Mechanism%29.ipynb#scrollTo=dnGwwU6p2Zfh
        x_embed = self.embedding_layer(input)
        x_embed_drop=self.dropout(x_embed)
        # Linear - ReLU - Linear
        # output and hn
        if self.model_type=="RNN":
            output,x_hidden=self.model(x_embed_drop)
        else:
            # i think for LSTM, we need to pass cell state
            # doc says it is a tuple
            # output,x_hidden,x_cell_state=self.model(x_embed_drop)
            # import pdb;pdb.set_trace()
            output,(x_hidden,x_cell_state)=self.model(x_embed_drop)
            # import pdb;pdb.set_trace()
            # using shared
            # hidden_ct=self.tanh(self.out(self.relu(self.linear(x_cell_state))))

        # are we using hidden? not sure?
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        Linear=self.linear(x_hidden)
        Relu=self.relu(Linear)
        x_relu=self.out(Relu)
        hidden=self.tanh(x_relu)
        # i think you should look at architecture given in pdf..
        if self.model_type=="LSTM":
            hidden=(hidden,x_cell_state)
        # there is no softmax here
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden
