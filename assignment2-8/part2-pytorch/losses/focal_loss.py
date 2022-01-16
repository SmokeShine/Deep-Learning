"""
Focal Loss Wrapper.  (c) 2021 Georgia Tech

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
import torch.nn.functional as F
import numpy as np

from collections import Counter

def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    # actually.. this is returning a disguised alpha
    # so it doesnot need loss function at all. 
    # I can write a simple unit test case for this.
    # okay.. unlike java.. this one does not generate test case on its own?
    # why not? test folder is recognized.. so unittest case is linked..
    # right click should have something
    # https://code.visualstudio.com/docs/python/testing
    # normalize weights to 1
    
    if beta<0 or beta >1:
        raise Exception("Beta Value Incorrect")
    # import pdb;pdb.set_trace()
    # this is a static value after one run

    # https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
    
    weights= [(1.-beta)/(1.-(beta**x)) for x in cls_num_list]
    # import pdb;pdb.set_trace()
    # check edsten..number may be incorrect
    per_cls_weights=[len(cls_num_list)*w/sum(weights) for w in weights]
    # import pdb;pdb.set_trace()
    # print(f"per_cls_weights:{per_cls_weights}")
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    # this has to be array
    
    return torch.from_numpy(np.array(per_cls_weights))


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        # sigmoid loss 
        # for softmax loss.. i will have to pass through exponential power for all the classes
        # which i think Andew NG said is very slow
        # https://www.youtube.com/watch?v=LLux1SW--oM
        # this is the discussion
        # where is the backward loss?
        # how was it happening before?
        # in part 1?
        # reweighting needs to be done here
        # softmax_ce had backward pass as well
        # take the unscaled raw value..  take the e. 
        # then use this e array to do division by sum
        # softmax backprop is not present
        # bro.. this is pytorch.. backward is pytorch magic
        # ok... so only forward pass then
        # some fresh air
        # input to the function.. is there a gamma? yes.. self.gamma
        # so that is fine
        # reweighting simply means.. i take 1-loss and take its power to gamma
        # so.. loss and gamma should be parameter for reweighting
        # no.. reweighting has num_classes as paraeter?
        # that was simple focal loss.. and that is loss..
        # no reweighted loss
        # equation 13.
        # changes.. first it is taking only number of classes as input
        # second...sum of losses.. that is focal loss
        # https://stackoverflow.com/questions/64751157/how-to-use-class-weights-with-focal-loss-in-pytorch-for-imbalanced-dataset-for-m
        # so.. lets think.. what can i take from pytorch softmax
        # first..softmax will already be a ratio of 1..so cannot calculate focal loss from it
        # ok..what else? binary cross entropy.. 
        # then? do the power thing.. and summation
        # then reweighting..

        # first, from labels.. calculate the frequency distribution
        # can hardcode to 10.. as it.. actually not.. it is not mnist
        # labels will have all values.. what is train does not have some values?
        # frequency_distribution=Counter(target)
        # frequency_distribution_sorted=sorted(frequency_distribution)
        # cls_num_list=[y for x,y in frequency_distribution_sorted.items()]
        # i think weight is beta
        # no.. function arguments are confusing.. gamma is beta
        # what is weight?
        # alpha is self.weight
        # self.weight=reweight(cls_num_list, self.gamma)
        # this is done in the main itself
        # Now.. chuck the vectorization part
        # is there a way to calculate softmax.. then I can simply multiply this value
        # what should be the size..
        # do i need to resize input?
        # this is a layer.
        # should this shape be 0 or 1?
        # it is not shape.. it is axis




        # z=nn.Softmax(dim=1)(input)  
        # z_log=torch.log(z)
        # added negative for loss. NLL loss put negative sign at the end by itself
        # logsoftmax=-nn.LogSoftmax(dim=1)(input)
        # is this log 10 or simple log
        # z loss will be scalar - nope.. when you calculate the overall loss, it will become scalar
        # import pdb;pdb.set_trace()
        # you didnot use target
        # so this was log 10.. not natural log
        # nn.function.log10()
        # target_one_hot=torch.nn.functional.one_hot(target,10)
        # element_wise_loss=logsoftmax
        # z_loss=nn.NLLLoss()(logsoftmax,target)
        # NLL is giving average loss
        # z loss needs to be batch size
        
        # should this only worry about probability of correct class?
        # i am not sure if this should be multiplied with one hot encoded target
        # non_prob=1-z
        # f_loss=torch.mul(torch.pow(non_prob,self.gamma),element_wise_loss)









        z=nn.Softmax(dim=1)(input)
        # dont use log 10 here.. else it wont match with log softmax
        z_log=-torch.log(z)
        
        # added negative for loss. NLL loss put negative sign at the end by itself
        logsoftmax=-nn.LogSoftmax(dim=1)(input)
        # is this log 10 or simple log
        # z loss will be scalar - nope.. when you calculate the overall loss, it will become scalar
        # import pdb;pdb.set_trace()
        # you didnot use target
        # so this was log 10.. not natural log
        # nn.function.log10()
        # target_one_hot=torch.nn.functional.one_hot(target,10)
        # element_wise_loss=logsoftmax
        z_loss=nn.NLLLoss()(nn.LogSoftmax(dim=1)(input),target)
        z_loss_no_reduce=nn.NLLLoss(reduction='none')(nn.LogSoftmax(dim=1) (input),target)

        # import pdb;pdb.set_trace()
        # NLL is giving average loss
        # z loss needs to be batch size
        
        # should this only worry about probability of correct class?
        # i am not sure if this should be multiplied with one hot encoded target
        non_prob=1-z
        # f_loss=torch.mul(torch.pow(non_prob,self.gamma),logsoftmax)

        ##CHecking with OH logic
        # import pdb;pdb.set_trace()
        pred_index=torch.argmax(z, axis=1)
        p_t=torch.zeros_like(pred_index).float()
        f_loss=torch.zeros_like(pred_index).float()
        weight_index_row=torch.zeros_like(pred_index).float()
        # import pdb;pdb.set_trace()
        for index_ in range(len(target)):
            pred_index_row=pred_index[index_]
            weight_index_row[index_]=self.weight[pred_index_row] if self.weight is not None else 1.
            # if pred_index_row==target[index_]:
            p_t[index_]=z[index_][target[index_]]
            # else:
            #     p_t[index_]=1-z[index_][target[index_]]
            
        
        # should there be a negative sign
        # This wont match with CE as CE takes sum of losses of all classes
        _loss=torch.mul(torch.pow(1-p_t,self.gamma),-torch.log(p_t))
        # import pdb;pdb.set_trace()
        # the weight should be for the particular correct index
        f_loss=torch.mul(weight_index_row,_loss)

        # f_loss=torch.tensor(f_loss)    
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        # this should match with normal calculation
        # torch_cel=torch.nn.CrossEntropyLoss()(input,target)
        # import pdb;pdb.set_trace()
        # f_loss_aggregated=f_loss.sum()
        # 32 would be batch size
        # import pdb;pdb.set_trace()
        # now what?
        # what does the formula say?
        # for each model.. multiply with  .. equation 8
        # do you have the class frequency from main?

        # per_cls_weights  << -- how to access this?
        # it is part of focal loss criterion? wtf?
        # so it is self.weight
        # what is z in the equation? z are predicted values
        # okay..changed the variable name
        # it is 32,10
        # there is log as well..that is fine.. standard formula..
        # only need to multiply with weights then
        # loss will be average loss? or sum?

        # element wise 
        # check the shape?
        # class_weighted_fl=0
        # for row in range(len(f_loss)):
        #     first=self.weight
        #     second=f_loss[row]
        #     # element wise multiplication
        #     # import pdb;pdb.set_trace()
        #     row_wise_sum=torch.mul(first,second).sum()
        #     # import pdb;pdb.set_trace()
        #     # print(row_wise_sum.item())
        #     class_weighted_fl+=row_wise_sum

        # # average loss as it is coming in batch
        # loss=class_weighted_fl/len(input)
    #     tensor([-1.1292, -0.4439, -0.1675, -0.0996, -0.1310,  0.0000,  0.0000,  0.0000,
    #     -0.1061,  0.0000], device='cuda:0', dtype=torch.float64,
    #    grad_fn=<SumBackward1>)
        # if self.weight is not None:
        #     loss=torch.mul(self.weight,f_loss).sum()
        # else:
        
        loss=f_loss.sum()
        # import pdb;pdb.set_trace()

        # loss.requires_grad = True
        # import pdb;pdb.set_trace()
        # https://zhuanlan.zhihu.com/p/28527749
        # May be.. do i need to put in cuda? that is a valid question
        # how will this work for test cases?
        # line 236.. it is already putting it in cuda.
        # what is happeing there?
        # reweight is supposed to return.. so that is correct
        # how do i test this part? is there a separate config file?
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss