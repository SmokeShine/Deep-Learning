"""
Transformer model.  (c) 2021 Georgia Tech

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
from torch import nn
import random

from torch.nn.modules import linear

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        ##############################################################################
        # TODO:
        #             #
        # You will need to use the max_length parameter above.                       #
        # This should take 1-2 lines.                                                #
        # Initialize the word embeddings before the positional encodings.            #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        # tbh.. i havent seen the lectures.. i havent read the paper..
        # why am i even opening this? for reading todo
        # weird.. what ?? i know what attention is.. but i dont seem to recall sin/cos
        # when is the next quiz date?/ why ? good to read instead of guessing
        # i am hungry.. need to eat something.. it is dineer time

        # max length should be for T
        # how to initialize them? torch random?
        # wtf is a bert embedding?
        # I have no clue .. no guidance to do this..canvas does not discuss this
        # OH is tangent to this assignment
        # may be it is concatenation? should be ..
        # but what is the dimension?
        # N,length of sentence, output dim
        # in the previous step?? where is the previous step?
        # may be i am dumb
        self.word_embedding=nn.Embedding(input_size,self.hidden_dim).to(self.device)
        # position encoding has to be loop.. i dont under these instructions..
        # making this non trainable
        self.position_embedding=nn.Embedding(self.max_length,self.hidden_dim).to(self.device)
        # is there a token embedding? looks like no
        # i am going to do dumb things because i want to forget about this assignment
        # like for loop
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k).to(self.device)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v).to(self.device)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q).to(self.device)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k).to(self.device)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v).to(self.device)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q).to(self.device)
        
        self.softmax = nn.Softmax(dim=2).to(self.device)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim).to(self.device)
        self.norm_mh = nn.LayerNorm(self.hidden_dim).to(self.device)

        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        # what kind of normalization are we talking about here?
        # there is norm_mh.. dont know what though
        # it is saying two transformer..
        # what is the issue? see.. transformer requires two transformers
        # to connect to each other.. 
        # that does not matke sense
        # all it needs is sen senpai..
        # need to check self. attributes
        # what is the most stupidest thing you can do?
        # add multiattention as a self object
        # what do i need to initialize here?
        # most likely a relu
        # may be i need to initialize linear layers? but what size? 
        #  from the diagram, tit looks like z1 and z2 are split again..
        # it is asking to add and normalize
        # followed by .. so that means two liear layer outputs needs to be added
        # and for numerical stability needs to be normalizes
        # but layer norm has shapes.. yes
        # hidden dimension
        # i need to look closely.. i am missing the complete story
        # most importantly, what is supposed to come as inputs?
        # is it ouput to deliverable 2.. which was residual and normalize?d
        # that would mean i am half way through
        # but then how did we end up with z1 and z2?
        # this is still the encoder side.. there would a decoder as well ..with masking
        # need to come again n that..
        # may be i need to reuse the linear layer.. but they are for k1 k2
        # this is different.
        # to i need to initialize bias as well?
        # it says element wise feed forward? element could be the tokens passed
        # one at a time
        # i am brute forcing.. this is not working
        # know the need to be symmetrical, i believe the linear
        # shape should be same
        # it has to be typo..
        # import pdb;pdb.set_trace()
        self.linear1 = nn.Linear(self.hidden_dim, self.dim_feedforward).to(self.device)
        self.relu=nn.ReLU().to(self.device)
        self.linear2 = nn.Linear(self.dim_feedforward, self.hidden_dim).to(self.device)
        # self.norm_layer = nn.LayerNorm(self.hidden_dim)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # what is the shape? is bias part of nn.linear? or separate?
        # it is two linear transformer.. not linear layers 
        # lol
        # self.weight1=nn.Linear()
        # may be it wants to initialize another layer norm.. that would mean dimension have changed..
        # but i dont know the output dim of linear
        # which would concatenate.. this link is broken
        # i am not returning z1 and z2.. there is no way? i can hack to split
        # may be it is better to put a trace
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.linear3 = nn.Linear(self.hidden_dim,self.output_size).to(self.device)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be normalized scores of shape (N,1).
        """

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling ClassificationTransformer class methods here.  #
        #############################################################################
        outputs = None
        # this is it.. this is the last part
        # i still havent done anything about masking..
        # actually, i havent done anything for decoder part..
        # what is this class? ClassificationTransformer
        # wtf.. there is nothing in search.. so it was never initiated..
        # how do i solve this..
        # the illustrated transformer..
        # is this a pytorch lingo that i am not aware.. and where was torchtext used?
        # may be in the notebook
        # no.. there is no method like this.. may be i need to instantiated this file
        # but the file name is transformer.py and the class is translator
        # i am stuck.. 
        # why my monitor does not have a usb port.. this is dumb
        # i am hungry..
        # this has to be an old artifact.. this has to be a trap
        # what is the dumbest thing you can do.. redo all the steps
        deliverable_1=self.embed(inputs=inputs).to(self.device)
        deliverable_2=self.multi_head_attention(inputs=deliverable_1).to(self.device)
        # lol.. devliverable 2 and 3 are linked
        # rip _suffix numbers
        deliverable_4=self.feedforward_layer(inputs=deliverable_2).to(self.device)
        # i have no f idea..what are doing here..
        outputs=self.final_layer(inputs=deliverable_4)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        embeddings = None
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################

        # what is happening here? input is an int tensor.. okay.. T is ? length of sentence?
        # it has to be padded.. else it cant come in batch
        # need to do simple pytorch embedding multiply.. 
        # is there self.embed object?
        # wtf is CLS token...this assignment is pure garbage
        # CLS is a classifier token..weird
        # 128 lenght?it is not divisible by 3..this is a shame that i have
        # to reverse engineer this as well
        # 64*2? hidden dim is 128
        
        inputs=inputs.to(self.device)
        
        we=self.word_embedding(inputs)
        # from edstem
        
        temp=torch.arange(0,self.max_length).expand_as(inputs).to(self.device)
        pe=self.position_embedding(temp)
        embeddings=we+pe
        # this is the dumbest thing i can do.. not sure what is input.. not sure
        # what is output.. but it should increase the ndims
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        outputs = None
        # sublayer x is weighted sum of attention
        # then there is residual sum
        # based on number of heads, we can have multiple attention matrices
        # input is 2,43,128
        # batch size of 2, 43 word length and 128 dimensionality 
        # what would you do?
        # self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        # self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        # self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # # Head #2
        # self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        # self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        # self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # i think input needs to be split into two
        # but that will break the dot product..no it is proection onto two
        # so this fine
        # why not check office hours?

        # what is residual - inputs only?
        
        query_vectors_1=self.q1(inputs)
        key_vectors_1=self.k1(inputs)
        value_vectors_1=self.v1(inputs)
        _,Nx,Dq=key_vectors_1.shape
        import math
        similarity_1=torch.bmm(query_vectors_1,key_vectors_1.permute(0,2,1))/math.sqrt(Dq)
        attention_1=self.softmax(similarity_1)
        output_vectors_1=attention_1@value_vectors_1

        query_vectors_2=self.q2(inputs)
        key_vectors_2=self.k2(inputs)
        value_vectors_2=self.v2(inputs)
        _,Nx,Dq=key_vectors_2.shape
        similarity_2=torch.bmm(query_vectors_2,key_vectors_2.permute(0,2,1))/math.sqrt(Dq)
        attention_2=self.softmax(similarity_2)
        output_vectors_2=attention_2@value_vectors_2
        # attention heads are concatenated one below another
        # import pdb;pdb.set_trace()
        # i have no interest left in this assignment.. this is boring work
        # it is 192.. matches with edtech posts now
        outputs_cat=self.attention_head_projection(torch.cat((output_vectors_1,output_vectors_2),2))
        # do i need to normalize?
        # https://jalammar.github.io/illustrated-transformer/
        # How do we do that? We concat the matrices then multiple them by an additional weights matrix WO.
        
        outputs=self.norm_mh(inputs+outputs_cat)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        outputs = None
        # import pdb;pdb.set_trace()
        # looks like a simple equation
        # why relu is required.. okay.. max o
        # what is weight 1?
        # nn.ReLU(inputs)
        linear_1=self.linear1(inputs)
        relu=self.relu(linear_1)
        linear_2=self.linear2(relu)
        outputs=self.norm_mh(inputs+linear_2) 
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code.                               #
        #############################################################################
        outputs = None
        # there is no help for this section.. ideally.. what were the encoding is,
        # we want to project it to?
        # wat is output size? token size?
        # no.. where is the decoder used?
        # output size and number of tokens will be different
        # yes.. 2,43,2..what is passed as 2?
        # import pdb;pdb.set_trace()
        outputs=self.linear3(inputs)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True