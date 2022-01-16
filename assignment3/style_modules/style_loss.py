import torch
import torch.nn as nn

class StyleLoss(nn.Module):
    def gram_matrix(self, features, normalize=True):
        """
            Compute the Gram matrix from features.

            Inputs:
            - features: PyTorch Variable of shape (N, C, H, W) giving features for
              a batch of N images.
            - normalize: optional, whether to normalize the Gram matrix
                If True, divide the Gram matrix by the number of neurons (H * W * C)

            Returns:
            - gram: PyTorch Variable of shape (N, C, C) giving the
              (optionally normalized) Gram matrices for the N input images.
            """
        ##############################################################################
        # TODO: Implement style loss function                                        #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        #                                                                            #
        # HINT: you may find torch.bmm() function is handy when it comes to process  #
        # matrix product in a batch. Please check the document about how to use it.  #
        ##############################################################################

        # pass
        # what is happening? are we given gram matrix or we need to calculate them?
        N, C, H, W=features.shape
        # unfortunately... this can be done only with changes in cublas environment behaviour
        # torch.use_deterministic_algorithms(True)
        # somehow, there is a difference of 0.99
        gram=torch.bmm(features.reshape(N,C,H*W),features.reshape(N,C,H*W).permute(0,2,1))
        if normalize:
              gram=torch.divide(gram,H * W * C)
        return gram
        # student_output.sum()
        # 2213.5144
        # correct.sum()
        # 2260.236
        # error
        # 0.9959817
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
    def forward(self, feats, style_layers, style_targets, style_weights):
        """
           Computes the style loss at a set of layers.

           Inputs:
           - feats: list of the features at every layer of the current image, as produced by
             the extract_features function.
           - style_layers: List of layer indices into feats giving the layers to include in the
             style loss.
           - style_targets: List of the same length as style_layers, where style_targets[i] is
             a PyTorch Variable giving the Gram matrix the source style image computed at
             layer style_layers[i].
           - style_weights: List of the same length as style_layers, where style_weights[i]
             is a scalar giving the weight for the style loss at layer style_layers[i].

           Returns:
           - style_loss: A PyTorch Variable holding a scalar giving the style loss.
           """

        ##############################################################################
        # TODO: Implement style loss function                                        #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        #                                                                            #
        # Hint:                                                                      #
        # you can do this with one for loop over the style layers, and should not be #
        # very much code (~5 lines). Please refer to the 'style_loss_test' for the   #
        # actual data structure.                                                     #
        #                                                                            #
        # You will need to use your gram_matrix function.                            #
        ##############################################################################

        # pass
        # lol this was left..
        # what is the expectation here
        # feats.. style layers..style targets..style weights.
        # most likely be list
        # there is tv loss as well.. ignore for the time being
        # one thing at a time
        # okay. everything is a list.. so most likely loop and sum
        # and return losses.. 
        # actually.. only forward.. why is there a forward here?
        # correct.. it says forward..but it is a loss
        # return loss
        # a simple way would be to zip all the layers and do weighted sum
        # do i need to normalize?
        # loop through all layers - gram matrix loss ..because it is only style
        loss=0.
        # may be this is required
        # style_loss.gram_matrix(feats[5].clone()).data.numpy()
        # do i need to normalize
        # i am pretty sure feats should have gram matrix but its length is not matching
        for i,val in enumerate(style_layers):
              # style layers is a list [1, 4, 6, 7]
              z=self.gram_matrix(feats[val].clone(),True)
              # feats and current shape are different.wtf
              # it is not that shape is different.. number of channels are different
              # style_targets need to use simple index
              loss=loss+style_weights[i]*torch.sum(torch.pow(torch.subtract(style_targets[i],z),2))
        return loss
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

