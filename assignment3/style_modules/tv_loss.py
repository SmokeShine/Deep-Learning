import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    def forward(self, img, tv_weight):
        """
            Compute total variation loss.

            Inputs:
            - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
            - tv_weight: Scalar giving the weight w_t to use for the TV loss.

            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img weighted by tv_weight.
            """

        ##############################################################################
        # TODO: Implement total variation loss function                              #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################

        # You should try to provide an efficient vectorized implementation.
        # what is coming.. it is an image
        # one image 3 channels
        # so? can i use any kernel?
        # x[:,1:] -x[:,:-1] should give difference is c dimension.
        # print(img.shape) torch.Size([1, 3, 192, 256])
        # diff1=img[:,1:] -img[:,:-1]
        # no.. one channel reduced.. channel should not drop
        # diff1=img[:,:,1:] -img[:,:,:-1]
        # torch.Size([1, 3, 191, 256])
        # why is this 191?
        # unfortunately, i cannot flatten it
        # https://discuss.pytorch.org/t/equivalent-function-like-numpy-diff-in-pytorch/35327/2
        diff_last_dimension=img[:,:,:,1:] -img[:,:,:,:-1]
        # shape didnot change
        # i have no idea what is happening
        # x = torch.tensor([1, 2, 4, 7, 0])
        # x_diff = x[1:] - x[:-1]
        # print(x_diff)
        # > tensor([ 1,  2,  3, -7])
        # okay.. i understood.. this is forward - current
        # okay..diff1.shape
        # torch.Size([1, 3, 192, 255])
        # img.shape
        # torch.Size([1, 3, 192, 256])
        # okay.. i think . why not call numpy then? additional space
        diff_second_last_dimension=img[:,:,1:,:] -img[:,:,:-1,:]
        # diff_last_dimension.shape
        # torch.Size([1, 3, 192, 255])
        # diff_second_last_dimension.shape
        # torch.Size([1, 3, 191, 256])
        # now..need to square and sum
        pow1=torch.pow(diff_last_dimension,2)
        pow2=torch.pow(diff_second_last_dimension,2)
        # need to sum across channels before multiplying with weights
        # shape is different
        # actually it is complete sum
        pow1_sum=pow1.sum()
        pow2_sum=pow2.sum()
        loss=tv_weight*(pow1_sum+pow2_sum)
        # what to return?
        # TV Loss Error is 0.289
        # TV Loss Error is 0.894
        # indexing was wrong
        return loss
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################