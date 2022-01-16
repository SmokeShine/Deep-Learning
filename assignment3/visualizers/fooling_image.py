import torch
from torch.autograd import Variable

class FoolingImage:
    def make_fooling_image(self, X, target_y, model):
        """
        Generate a fooling image that is close to X, but that the model classifies
        as target_y.

        Inputs:
        - X: Input image; Tensor of shape (1, 3, 224, 224)
        - target_y: An integer in the range [0, 1000)
        - model: A pretrained CNN

        Returns:
        - X_fooling: An image that is close to X, but that is classifed as target_y
        by the model.
        """

        model.eval()

        # Initialize our fooling image to the input image, and wrap it in a Variable.
        X_fooling = X.clone()
        X_fooling_var = Variable(X_fooling, requires_grad=True)

        # We will fix these parameters for everyone so that there will be
        # comparable outputs

        learning_rate = 10
        max_iter = 100  # maximum number of iterations

        for it in range(max_iter):

            ##############################################################################
            # TODO: Generate a fooling image X_fooling that the model will classify as   #
            # the class target_y. You should perform gradient ascent on the score of the #
            # target class, stopping when the model is fooled.                           #
            # When computing an update step, first normalize the gradient:               #
            #   dX = learning_rate * g / ||g||_2                                         #
            #                                                                            #
            # Inside of this loop, write the update rule.                                #
            #                                                                            #
            # HINT:                                                                      #
            # You can print your progress (current prediction and its confidence score)  #
            # over iterations to check your gradient ascent progress.                    #
            ##############################################################################
            
            # forgot everything because of assignment
            # what is g? g should be gradient
            # do i need to perform forward and backward here?
            # model parameter is passed. so model forward and backward should be present
            
            output=model.forward(X_fooling_var)
            # loss should be? again same as saliency?
            # this loss should not be for target class
            # target_y is the wrong class
            # so len wont work
            # does the class start from 0 or 1?
            # integer [0,1000)
            # if max value is target_y then break
            # else do backward
            # what is loss value?
            # Given an image and a target class, we can perform gradient ascent over
            # the image to maximize the target class,
            if output.argmax()==target_y:
                # i cannot see the knot that is present in TA output
                # there is a bug somewhere
                break
            else:
                # cant use range as target_y is int
                loss_individual=output[0][target_y]
                # this is not required but anyhow
                loss=loss_individual.sum()
                # perform backward pass
                loss.backward()
                # get the gradients
                X_var_grad=X_fooling_var.grad.data
                # does torch has a function for the second part?
                # l2 norm is for a vector.. not a matrix
                # largest eigen value of A.T * T
                # good luck solving this
                # https://pytorch.org/docs/stable/generated/torch.linalg.norm.html#torch.linalg.norm
                # Computes a vector or matrix norm.
                # so there should a modifier for matrix
                # torch.linalg.matrix_norm
                # how to know if this is l2 norm
                # l2_norm=torch.linalg.matrix_norm(X_var_grad).sum()
                # this was 3D; could be 1 for each channel
                # https://pytorch.org/docs/stable/generated/torch.norm.html
                # frobenius norm
                # l2_norm=X_var_grad.pow(2).sum(dim=1).sqrt()
                # https://stackoverflow.com/questions/68489765/what-is-the-correct-way-to-calculate-the-norm-1-norm-and-2-norm-of-vectors-in
                # pytorch api is very unintuitive
                # i think it is better to write the function by definition itself
                l2_norm=X_var_grad.pow(2).sum().sqrt()
                # this return a single value
                # l2_norm
                # tensor(2.5593)
                # torch.linalg.matrix_norm(X_var_grad)
                # tensor([[1.4618, 1.8751, 0.9473]])
                # i dont understand the transformation
                # it is not sum, it is not mean,it is not max..wtf is it then?
                # read the source code.. 
                X_fooling_var.data=X_fooling_var.data+learning_rate*X_var_grad/l2_norm
                # dont know why people are saying zero grad is necessary. it will be overwritten
                X_fooling_var.grad.data.zero_()
                # this does not make sense, the knot is visible now :X
                # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903
                # Every time a variable is back propogated through, 
                # the gradient will be accumulated instead of being replaced. 
                # so, it was doing some accumulation.. what does that mean?
                # why accumulation is wrong here? so.. it is not updating.. so?
                # norm will keep on increasing?
            # no need for abs here
            ##############################################################################
            #                             END OF YOUR CODE                               #
            ##############################################################################

        X_fooling = X_fooling_var.data

        return X_fooling
