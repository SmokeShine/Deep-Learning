import numpy as np
# added for np.square
class _BaseOptimizer:
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def update(self, model):
        pass

    def apply_regularization(self, model):
        '''
        Apply L2 penalty to the model. Update the gradient dictionary in the model
        :param model: The model with gradients
        :return: None, but the gradient dictionary of the model should be updated
        '''

        #############################################################################
        # TODO:                                                                     #
        #    1) Apply L2 penalty to model weights based on the regularization       #
        #       coefficient                                                         #
        #############################################################################
        # reg_diff = w_grad_reg - w_grad
        # expected_diff = model.weights['W1'] * optimizer.reg
        for feature in model.gradients:
            # Remember, you may NOT want to apply regularization on bias terms!
            # which is the bias term?
            if 'b' not in feature:
                model.gradients[feature]=model.weights[feature]*self.reg + model.gradients[feature]
        # model.gradients['W1']=model.weights['W1']*self.reg + model.gradients['W1']
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################