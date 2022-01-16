"""
SGD Optimizer.  (c) 2021 Georgia Tech

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

from ._base_optimizer import _BaseOptimizer


class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

    def update(self, model):
        """
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        """
        self.apply_regularization(model)
        # this is only increasing the gradient and it overwrote the values
        # so the learning rate will be applied on this as well
        # print("========SGD===========")
        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                # this is optimizer.. so.. we need to do what?
                # call backpropation?
                # okay.. what is the testcase looking for? any module?
                # any function call?
                # what is SGD with momentum? 
                # it is mainly about using past values of loss function? loss function?
                # past gradients.. and exponentially summing them?
                # w- learning rate* dw this is sgd
                # do i need to care about batch size? good question
                # i think all the gradients are coming in batch only.. there is N at the start
                # so.. need to take average value .. m or 2m?
                # it is SGD ..not mini batch gradient descent..
                # it should be single row
                # okay.. that is sorted..
                # can you even recall the formula? weight - learning rate * dv
                # where dv is adding a momentum on the current
                # so initial v will be zero. and then ? no bias correction
                # that is in rms prop
                # is momentum applied to both bias and weights?
                # stored exact value of v.. then exponentially add to get new value?
                # okay.. use the formula.. and store.. store where?
                # self.grad_tracker is a dictionary..
                # thinking.. do i need to update the value of dictionary for all
                # values?
                # update should updated the weights
                # dictionary needs to hold a w and b as keys
                # and velocity components in array
                # what? velocity will be diffeent for each weight and bias 
                # then each weight should be key and we can loop
                # optimizer has to access model function and store a small cache
                # okay.. it is stored in a different .. there is idx from model.modules
                # i think it is accessing files on the disk
                # there is no import.. relative..that will be difficult to debug then
                # it is __dict__
                # there is nothing called m.dw in the model parameters
                # updated forward and backward.
                # it is manual forward and push in the test and not a loop
                #
                # grad tracker.. most likely requires a for loop..but it is a not a list..
                # cant reverse engineer the test case. it is reading from binary. 

                # need to move faster.. hasnt even started pytorch..

                # may be loop is not required?
                # In practice, it is common to use a momentum term in SGD for better convergence.
                # Specifically, we introduce a new velocity term vt and the update rule is as
                # follows:
                # where β denotes the momentum coefficient and η denotes the learning rate
                # where is learning rate? self.learning rate?
                # beta is momentum 0.9
                # so each module was dw and db .. so they can be stored.
                # as they are already initialized, I can call those values
                # the for loop is same.
                # print(f"{m} weight")
                # try:
                #     print(f"weight:{self.grad_tracker[idx]['dw'].sum()}")
                # except:
                #     pass
                # self.grad_tracker[idx]['dw']=self.momentum*self.grad_tracker[idx]['dw']-self.learning_rate*m.dw
                # m.weight=m.weight+self.grad_tracker[idx]['dw']

                self.grad_tracker[idx]['dw']=self.momentum*self.grad_tracker[idx]['dw']+m.dw
                m.weight=m.weight - self.learning_rate * self.grad_tracker[idx]['dw']
                
                # m.weight=m.weight-self.learning_rate*m.dw
                # print(f"After update dw:{self.grad_tracker[idx]['dw'].sum()}")
                # pass
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                # same...?? right?
                # Finished epoch 0 / 10: cost 2.306153, train: 0.180000, lr 1.000000e-04
                # Finished epoch 1 / 10: cost 2.251948, train: 0.320000, lr 9.500000e-05
                # Finished epoch 2 / 10: cost 2.159310, train: 0.380000, lr 9.025000e-05
                # try:
                #     print(f"bias:{self.grad_tracker[idx]['db'].sum()}")
                # except:
                #     pass
                # self.grad_tracker[idx]['db']=self.momentum*self.grad_tracker[idx]['db']-self.learning_rate*m.db
                # m.bias=m.bias+self.grad_tracker[idx]['db']

                self.grad_tracker[idx]['db']=self.momentum*self.grad_tracker[idx]['db']+m.db
                m.bias=m.bias - self.learning_rate * self.grad_tracker[idx]['db']
                # m.bias=m.bias-self.learning_rate*m.db
                # print(f"After update db:{self.grad_tracker[idx]['db'].sum()}")

            # import numpy as np
            # np.save(f'tests/debug/w_{idx}.npy',m.weight)
            # np.save(f'tests/debug/b_{idx}.npy',m.bias)
            # np.save(f'tests/debug/dw_{idx}.npy',m.dw)
            # np.save(f'tests/debug/db_{idx}.npy',m.db)
                # pass
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
