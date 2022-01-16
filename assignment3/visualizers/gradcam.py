import torch
from torch.autograd import Function as TorchFunc
import numpy as np
from PIL import Image


# The ’deconvolution’ is equivalent to a backward pass through the network, except that 
# when propagating through a nonlinearity, its gradient is solely computed based on the 
# top gradient signal, ignoring the bottom input. In case of the ReLU nonlinearity this 
# amounts to setting to zero certain entries based on the top gradient. We propose to 
# combine these two methods: rather than masking out values corresponding to negative 
# entries of the top gradient (’deconvnet’) or bottom data (backpropagation), we mask 
# out the values for which at least one of these values is negative.

class CustomReLU(TorchFunc):
    """
    Define the custom change to the standard ReLU function necessary to perform guided backpropagation.
    We have already implemented the forward pass for you, as this is the same as a normal ReLU function.
    """

    @staticmethod
    def forward(self, x):
        output = torch.addcmul(torch.zeros(x.size()), x, (x > 0).type_as(x))
        self.save_for_backward(x, output)
        # print(x.shape)
        return output

    @staticmethod
    def backward(self, y):
        ##############################################################################
        # TODO: Implement this function. Perform a backwards pass as described in    #
        # the guided backprop paper ( there is also a brief description at the top   #
        # of this page).                                                             #
        # Note: torch.addcmul might be useful, and you can access  the input/output  #
        # from the forward pass with self.saved_tensors.                             #
        ##############################################################################
        # pass
        # https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
        x,output=self.saved_tensors
        # what is y? dout?
        y_clone=y.clone()
        y_clone[x<0]=0.
        y_clone[y<0]=0.
        # what is it supposed to return?
        # gradient most likely
        return y_clone
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################


class GradCam:
    def guided_backprop(self, X_tensor, y_tensor, gc_model):
        """
        Compute a guided backprop visualization using gc_model for images X_tensor and 
        labels y_tensor.

        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the guided backprop.

        Returns:
        - guided backprop: A numpy of shape (N, H, W, 3) giving the guided backprop for 
        the input images.
        """

        # Thanks to Farrukh Rahman (Fall 2020) for pointing out that Squeezenet has
        #  some of it's ReLU modules as submodules of 'Fire' modules
        #  
        for param in gc_model.parameters():
            param.requires_grad = True

        for idx, module in gc_model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                gc_model.features._modules[idx] = CustomReLU.apply
            elif module.__class__.__name__ == 'Fire':
                for idx_c, child in gc_model.features[int(idx)].named_children():
                    if child.__class__.__name__ == 'ReLU':
                        gc_model.features[int(idx)]._modules[idx_c] = CustomReLU.apply
        ##############################################################################
        # TODO: Implement guided backprop as described in paper.                     #
        # (Hint): Now that you have implemented the custom ReLU function, this       #
        # method will be similar to a single training iteration.                     #
        #                                                                            #
        # Also note that the output of this function is a numpy.                     #
        ##############################################################################
        
        output=gc_model(X_tensor)
        loss_individual=output.gather(1,y_tensor.reshape(len(y_tensor),1))
        loss=loss_individual.sum()
        # perform backward pass
        loss.backward()
        # gc_model.backward()
        # pass
        # gbp_result
        # is it loss function?
        input_grad=X_tensor.grad.data.permute(0,2,3,1)
        # check if type is numpy
        return input_grad.detach().numpy()
        # it is supposed to return something
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

    def grad_cam(self, X_tensor, y_tensor, gc_model):
        """
        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the gradcam.
        """
        conv_module = gc_model.features[12]
        self.gradient_value = None  # Stores gradient of the module you chose above during a backwards pass.
        self.activation_value = None  # Stores the activation of the module you chose above during a forwards pass.

        def gradient_hook(a, b, gradient):
            self.gradient_value = gradient[0]

        def activation_hook(a, b, activation):
            self.activation_value = activation

        conv_module.register_forward_hook(activation_hook)
        conv_module.register_backward_hook(gradient_hook)
        ##############################################################################
        # TODO: Implement GradCam as described in paper.                             #
        #                                                                            #
        # Compute a gradcam visualization using gc_model and convolution layer as    #
        # conv_module for images X_tensor and labels y_tensor.                       #
        #                                                                            #
        # Return:                                                                    #
        # If the activation map of the convolution layer we are using is (K, K) ,    #
        # student code should end with assigning a numpy of shape (N, K, K) to       #
        # a variable 'cam'. Instructor code would then take care of rescaling it     #
        # back                                                                       #
        ##############################################################################
        cam=None
        output=gc_model(X_tensor)
        loss_individual=output.gather(1,y_tensor.reshape(len(y_tensor),1))
        loss=loss_individual.sum()
        # perform backward pass
        loss.backward()
        # should this be vector or matrix?
        # alpha=self.gradient_value.sum(axis=[0,2,3])/(len(self.gradient_value)*len(self.gradient_value[0][0].flatten()))
        # alpha=self.gradient_value.sum(axis=[2,3])/(len(self.gradient_value[0][0].flatten()))
        alpha=self.gradient_value.mean(axis=[2,3])
        # alpha=self.gradient_value.sum(axis=-1).sum(axis=-1)/(len(self.gradient_value[0][0].flatten()))
        # print(self.gradient_value.shape)
        # print(alpha.shape)
        # this should be mean
        weighted_sum=torch.einsum ('ijkl, ij -> ijkl', self.activation_value, alpha)
        with torch.no_grad():
            cam=torch.relu(weighted_sum.sum(axis=1)).detach().numpy()
        # https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html
        # took the easy way out.. storing in self.gradient_value

        # now.. use the hook to get the values.
        # need to sum the values..
        # and then do a weighted sum..
        # pass through relu
        # this is very similar to the pytorch code i wrote. 
        # hook saves a dictionary for dx at that layer
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

        # Rescale GradCam output to fit image.
        cam_scaled = []
        for i in range(cam.shape[0]):
            cam_scaled.append(np.array(Image.fromarray(cam[i]).resize(X_tensor[i, 0, :, :].shape, Image.BICUBIC)))
        cam = np.array(cam_scaled)
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam
