from ._base_optimizer import _BaseOptimizer
import numpy as np
class SGD(_BaseOptimizer):
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        super().__init__(learning_rate, reg)

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)
        #############################################################################
        # TODO:                                                                     #
        #    1) Update model weights based on the learning rate and gradients       #
        #############################################################################
        for feature in model.weights.keys():
            model.weights[feature]=model.weights[feature]-self.learning_rate*model.gradients[feature]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
