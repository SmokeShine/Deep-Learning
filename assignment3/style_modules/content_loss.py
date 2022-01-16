import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def forward(self, content_weight, content_current, content_original):
        """
            Compute the content loss for style transfer.

            Inputs:
            - content_weight: Scalar giving the weighting for the content loss.
            - content_current: features of the current image; this is a PyTorch Tensor of shape
              (1, C_l, H_l, W_l).
            - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

            Returns:
            - scalar content loss
            """

        ##############################################################################
        # TODO: Implement content loss function                                      #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################

        # pass
        # what is happening.. content weight, content current and content original
        # different from x and  target y.. does this mean, we will have to set the var variable as well
        # need to return a scalar loss
        # content weight is just a multiplier
        # content current is the image.. there is no model here
        # what is the difference between current image and content image
        # may be current image may be ? there is no loop.. 
        # there is a test case.. read the test case. there is a formula as well
        # there should be loop outside.. as this is for a single feature..and there is no
        # constraint on the depth chosen.. so most likely it is adding all of them
        # style transfer has a test function
        # what is it doing? it is checking the output with a binary..loss will be zero 
        # if the activations maps are same
        # https://www.bing.com/search?q=torch.math+&qs=n&form=QBRE&sp=-1&pq=&sc=0-0&sk=&cvid=514E5CBA2B5A457A96535490D11D66FE
        loss=content_weight*torch.sum(torch.pow(torch.subtract(content_current,content_original),2))
        return loss
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

