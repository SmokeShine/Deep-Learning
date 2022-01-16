"""
2d Convolution Module.  (c) 2021 Georgia Tech

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


class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,new_seed=True):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None
        
        self._init_weights(new_seed)

    def _init_weights(self,new_seed=True):
        if new_seed==True:
            np.random.seed(1024)
            self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)            
        else:
            self.weight=None
            
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        # watched the videos
        # need to use numpy strides.. but stride depend on system float value
        # a.itemsize
        # need to slide the array 
        # cant use row?columns? would require hardcoding..
        # can be done in loop 
        # try to use office hour method..useful later
        # padding as well? wtf
        # x is 4,3,5,5. 4 rows..3 channels.. size of 5,5.. symmetrical
        # filter size? i think it is kernel size 3,3
        # why padding is required? for passig test case
        # becasue padding is non zero
        # https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
        
        # what if padding size is 0?this implementation is wrong
        if self.padding!=0:
            only_h_w=((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding))
            padded=np.pad(x,only_h_w)
        else:
            padded=x.copy()
        # how will the shape change? 6,5,7,7?
        # something is off.. number of rows increased 
        # padded is now 4,3,7,7
        # sum should be same
        # padded[0][0].sum()
        # 7.221162858969281
        # x[0][0].sum()
        # 7.221162858969282
        # now what?stride logic is very difficult to understand
        # it says to find the shape first
        N, C, H_padded, W_padded=padded.shape
        # filter size.. kernel is symmetrical
        # h_new=((H_padded-self.kernel_size)//self.stride)+1
        # same formula that is shown in the OH video.
        # w_new=((W_padded-self.kernel_size)//self.stride)+1
        # now what.. there is this ugly stride formula
        # https://bluejeans.com/playback/s/DZiAnMlhYvOQTWC6m8zfHfO9agrbh8nuvwDfFoAGAIQPJ6Huo7r0d2U3nxmi8HxF
        # what is A?unpadded or padded
        # what is 3? number of channels?in channel or out channel
        # brute force..see which one runs
        # what is s_h? padded or unpadded?
        # this is 4d.. video has 3d on the first value
        # N will still remain a problem
        # s_c,s_h,s_w=padded[0].strides
        # (1176, 392, 56, 8)
        # B=np.lib.stride_tricks.as_strided(padded[0],
        # shape=(h_new,w_new,C,self.kernel_size,self.kernel_size),
        # strides=(s_h*self.stride,s_w*self.stride,s_c,s_h,s_w),
        # writeable=False)
        # np.tensordot(padded,B,axes=3)
        # video says this will work.. A is nchw and B is chwk
        # A is an image.. what is N?
        # padded is 4377
        # B is 3,3,3,3,3,3
        # 2,3,45,40 vs 22,19,3,3,3
        # watch the video again.. intentional knowledge skip
        # lol.. kernel has a bias as well

        # unfortunately.. vectorization is not possible with so less of a time.. np function looks a monster
        # and everybody is going back to for loops

        # What is the logic? 
        # conv filter needs to be rotated 180 degrees
        # and element wise multiplication
        # what is the issue.. number of patches to be generated.. 
        # how to test.. actually.. even 1 conv can be tested..
        # but what is in the test case?
        # (?,number of images,number of filter, output image size, output image size) 
        # output dimension
        # in channel 3
        # out channel 3
        # kernel size 4
        # stride 2
        # padding 1
        # 2*2 filter..* 3 (number of filters) for 2 images

        # x is number of images, number of channels, H and Width
        # w is number of filters,number of channels, kernel size row,column

        # check later for int and float
        H_new=((H_padded-self.kernel_size)//self.stride) +1
        W_new=((W_padded-self.kernel_size)//self.stride) +1
        # (N, self.out_channels, H', W')
        out=np.zeros([N, self.out_channels, H_new, W_new])
        # shape is sorted. can you manually do conv and test?
        for image_number in range(N):
            image=padded[image_number]
            # for channel_number in C:
                # image_channel=image[channel_number]
                # rotate 180 degrees
                # is there a numpy function?
                # https://numpy.org/doc/stable/reference/generated/numpy.rot90.html
                # image_channel_180=np.rot90(np.rot90(image_channel))
                # temp[channel_number]=image_channel_180
                # need to rotate kernel ...not image
            for filter_number in range(len(self.weight)):
                filter_weights=self.weight[filter_number]
                filter_copy=filter_weights.copy()
                for channel_number in range(C):
                    filter_channel=filter_copy[channel_number]
                    filter_channel_180=filter_channel
                    # as per the test case, it is cross correlation and not convolution
                    # filter_channel_180=np.rot90(np.rot90(filter_channel))
                    filter_copy[channel_number]=filter_channel_180
                    # now the filter is rotated 180 degrees
                # now cross correlation for this filter
                # but all combinations of images needs to be generated
                _,row_,column_=image.shape
                # why multiplication is required?
                # because by this, we are saying we want H_new dimensions
                # for i in range(0,self.stride*H_new,self.stride):
                for i in range(0,self.stride*H_new,self.stride):
                    for j in range(0,self.stride*W_new,self.stride):
                        # should this be stride or kernel size?
                        # is this 3d or 2d?
                        patch=image[:,i:i+self.kernel_size,j:j+self.kernel_size]
                        # if patch.shape==filter_copy.shape:
                        # https://stackoverflow.com/questions/34003573/numpy-3d-dot-product
                        if len(filter_copy.flatten())!=len(patch.flatten()):
                            # the issue is with the number of channels
                            x=1
                        image_conv=filter_copy.flatten().dot(patch.flatten())+self.bias[filter_number]
                        # image_conv=np.sum(np.multiply(filter_copy,patch))+self.bias[filter_number]
                        # where to save it?
                        # (N, self.out_channels, H', W')
                        # fix for zero index
                        # this is pasting in the wrong cell
                        out[image_number,filter_number,i//self.stride,j//self.stride]=image_conv

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        # out=np.expand_dims(out,0)
        self.cache = x
        # print(self.padding)
        return out

    # def backward(self, dout):
    #     """
    #     The backward pass of convolution
    #     :param dout: upstream gradients
    #     :return: nothing but dx, dw, and db of self should be updated
    #     """
    #     x = self.cache
    #     #############################################################################
    #     # TODO: Implement the convolution backward pass.                            #
    #     # Hint:                                                                     #
    #     #       1) You may implement the convolution with loops                     #
    #     #       2) don't forget padding when computing dx                           #
    #     #############################################################################
    #     #4355 
    #     # it is unpadded image
    #     # if self.padding!=0:
    #     #     only_h_w=((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding))
    #     #     padded=np.pad(x,only_h_w)
    #     # else:
    #     #     padded=x

    #     # now.. now what?
    #     # do you even know what to do in backpropogation of convolution?
    #     # convolution becomes cross cross correlation
    #     # not enough to solve
    #     # dout is the upsteam gradient - same pixel is used across the image
    #     # same filter is used all pixels of image
    #     # first pixel of next output layer is influenced
    #     # by the ?
    #     # 3 channels...image patch of filter size

    #     # first flip the kernel by 180
    #     # https://numpy.org/doc/stable/reference/generated/numpy.rot90.html
    #     N, C, H, W=x.shape
    #     # it may not work.. but rotate all the filters by 180. then 
    #     # do conv forward on all four images
    #     # https://numpy.org/doc/stable/reference/generated/numpy.rot90.html
    #     flipped_weights=self.weight.copy()
    #     bias=np.zeros(len(self.weight))
    #     for filter_number in range(len(self.weight)):
    #         filter_weights=self.weight[filter_number]
    #         filter_copy=filter_weights.copy()
    #         for channel_number in range(C):
    #             filter_channel=filter_copy[channel_number]
    #             filter_channel_180=filter_channel
    #             filter_channel_180=np.rot90(np.rot90(filter_channel))
    #             filter_copy[channel_number]=filter_channel_180
    #             # Now.. this to conv forward with a padded dout
    #             # what about dout channels
    #         flipped_weights[filter_number]=filter_copy
    #     # dout is 4255
    #     # only_h_w=((0,0),(0,0),(self.kernel_size-1,self.kernel_size-1),(self.kernel_size-1,self.kernel_size-1))
    #     # dout_padded=np.pad(dout,only_h_w)                    
    #     # call conv forward
    #     # do not call test function..it may not exist in grader
    #     # what will be the output channel? how to verify it?
    #     # out_channel already holds information about number of filters
    #     # i think out will become in and vice versa
    #     # flipped weights is 2333
    #     # loop through all filters and add derivatives
    #     all_derivatives=np.zeros([N,C,H,W])
    #     image_derivatives=[]
    #     # may be wrong formula
    #     # need to verify this shape.. cant recall
    #     # should this be same as self.padding?
    #     # sometimes padding is required..sometimes it is not
    #     # not sure about this padding
    #     # padding_=1
    #     padding_=(H*self.stride-1+self.kernel_size-H)//2
    #     # padding_=(H*self.stride-1+self.kernel_size-H)//2
    #     # padding_=(H*self.stride-self.stride+self.kernel_size-H)//2
    #     for image_number in range(N):
    #         image_derivatives_sum=[]
    #         image_derivatives=[]
    #         for filter_number in range(dout.shape[1]):                
    #             dout_patch=dout[image_number][filter_number]
    #             r,c=dout_patch.shape
    #             # https://stackoverflow.com/questions/27125959/numpy-array-insert-alternate-rows-of-zeros/27126043
    #             # https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-fb2f2efc4faa
    #             # stride is always 1. so this wont help
    #             # so the issue is not here
    #             if self.stride>1:
    #                 dout_patch_expanded=np.zeros([2*r,2*c])
    #                 dout_patch_expanded[::2, ::2] = dout_patch 
    #                 padding_=1
    #             else:
    #                 dout_patch_expanded=dout_patch
    #             dout_patch_expanded=np.expand_dims(dout_patch_expanded,0)
    #             dout_patch_expanded=np.expand_dims(dout_patch_expanded,0)
    #             # need to expand dim 
    #             channel_derivatives=[]
    #             # causing error
    #             # for channel_number in range(len(flipped_weights[filter_number])):
    #             for channel_number in range(len(flipped_weights[filter_number])):
    #                 flipped_weights_channel=flipped_weights[filter_number][channel_number]
    #                 flipped_weights_channel=np.expand_dims(flipped_weights_channel,0)
    #                 # making the weights 4d 
    #                 flipped_weights_channel=np.expand_dims(flipped_weights_channel,0)
    #                 conv = Conv2D(in_channels=1,
    #                             out_channels=1,
    #                             kernel_size=self.kernel_size,
    #                             stride=1, #hardcoding to 1
    #                             padding=padding_)
    #                             # not sure about this. padding along the edges
    #                             # https://hideyukiinada.github.io/cnn_backprop_strides2.html
    #                 # what about weights.how to transfer that
    #                 conv.weight = flipped_weights_channel
    #                 # no correction this time
    #                 # conv.bias = 0
    #                 # dout_patch - need to insert zeros in between.. what about stride?
    #                 # no stride i think
                    
    #                 channel_derivative=conv.forward(dout_patch_expanded)
    #                 # this is 5 dimensional
    #                 # this is the culprit
    #                 # this is the culprit again. this is looking for padded size
    #                 # why is this coming of a different shape?
    #                 # padded wont be updated. cutting off
    #                 # import pdb;pdb.set_trace()
    #                 # cutting cannot be based on padding
    #                 # try:
    #                 # why always 1?
    #                 squeezed_channel_derivative=channel_derivative[0][0][0][1:1+H,1:1+W].reshape(H,W)
    #                 # except:
    #                 #     squeezed_channel_derivative=channel_derivative[0][0][0].reshape(H,W)
    #                 channel_derivatives.append(squeezed_channel_derivative)
    #             # how is the sum happening.. shape should remain the same as input image
    #             channel_reshape=np.array(channel_derivatives).reshape(C,H,W) #removed sum
    #             image_derivatives.append(channel_reshape)
    #             # image derivatives hold for all the filters
    #         # sum the derivatives from all the filters
    #         # axis=1 means sum across the filters
    #         # axis=0 when both filter and channel are more than 1
    #         # if filters>1, then axis=1

    #         np_image_derivatives=np.array(image_derivatives).reshape(dout.shape[1],C,H,W).sum(axis=0)
    #         # this is summing across images??
    #         # image_derivatives_sum=np.sum(np_image_derivatives,axis=1)
    #         # how to store if this works correctly?
    #         # all_derivatives[image_number]=image_derivatives_sum
    #         all_derivatives[image_number]=np_image_derivatives
    #     # treat each channel as a separate image and doing some depth wise
    #     # array was adding first element
    #     self.dx=all_derivatives

    #     # db 
    #     # sum the first row from the dout 4 segments
    #     # then sum vertically; this will flip the axis
    #     # so rotate again
            

    #     self.db=dout.sum(axis=3).sum(axis=2).sum(axis=0)

    #     # now dw
    #     # https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
    #     # https://github.com/neuron-whisperer/cnn-numpy/blob/master/cnn_numpy.py
    #     # We can notice that dw is a convolution of the input x with a filter dy
    #     # ok..so similar to dx.. but what about the shape? dy is the filter? rotate dout twice?
    #     # no .. i dont
    #     # may be pad?
    #     # padding_=(H*self.stride-self.stride+self.kernel_size-H)//2
    #     # this is incorrect - failing python train.py
    #     s=self.stride
    #     k=len(self.weight[0][0])
    #     f=len(dout[0][0])
    #     n=H
    #     # always pad with 1?
    #     # padding_=(s*k-s+f-n)//2
    #     padding_=(s*k-s+f-n)//2
    #     # padding_=(H*self.stride-self.stride+len(dout[image_number])-self.kernel_size)//2
        
    #     channel_derivatives=[]
    #     # some combinations are not running
    #     filter_channel_derivatives=[]
    #     filter_derivatives=[]
    #     w_sum=np.zeros([dout.shape[1],C,self.kernel_size,self.kernel_size])
    #     for filter_number in range(dout.shape[1]):
    #         filter_derivative=[]
    #         for image_number in range(N):    
    #             conv=Conv2D(in_channels=1,
    #                                 out_channels=1,
    #                                 kernel_size=len(dout[0][0]),
    #                                 stride=self.stride,
    #                                 padding=padding_)
    #             conv.weight = np.expand_dims(np.expand_dims(dout[image_number][filter_number],0),0)
    #                         # no correction this time
    #             # conv.bias = np.zeros(len(dout))
    #             channel_derivatives=[]
    #             for channel_number in range(C):
    #                 img=np.expand_dims(np.expand_dims(x[image_number][channel_number],0),0)
    #                 channel_derivative=conv.forward(img)
    #                 channel_derivative=channel_derivative.reshape(self.weight[0][0].shape)
    #                 channel_derivatives.append(channel_derivative)
    #             filter_derivative.append(channel_derivatives)    
    #             # w_sum[filter_number][channel_number]=np.array(filter_derivative).sum(axis=1)
                
    #             # w_sum[filter_number][channel_number]=np.array(filter_derivative).sum(axis=0)
    #         # filter_derivatives.append(filter_derivative)
    #         w_sum[filter_number]=np.array(filter_derivative).sum(axis=0)
    #         filter_derivatives=[]

    #     self.dw=w_sum

        
        # self.dw=dw
        # line 171 has a bug
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    def backward(self, dout):
        x=self.cache
        N, C, H, W=x.shape
        dXpadded=np.zeros([N,C,H+2*self.padding,W+2*self.padding])

        # https://edstem.org/us/courses/8409/discussion/651409
        for obs in range(N):
            for c in range(self.weight.shape[0]):
                for h in range(dout.shape[-2]):
                    for w in range(dout.shape[-1]):
                        # will this require any padding?
                        starting_row_index=h*self.stride
                        ending_row_index=starting_row_index+self.kernel_size
                        starting_column_index=w*self.stride
                        ending_column_index=starting_column_index+self.kernel_size
                        try:
                            doutval=dout[obs,c,h,w]
                        except:
                            z=1
                        # this should be scalar
                        current_filter=self.weight[c,:]
                        # shape of 3,3,3
                        Result=current_filter*doutval
                        # need to insert this
                        try:
                            dXpadded[obs, :, starting_row_index:ending_row_index, starting_column_index:ending_column_index]+=Result
                        except:
                            z=1
        # extracting dx from dxpadded
        if self.padding!=0:
            dx = dXpadded[:, : , self.padding:- self.padding, self.padding:-self.padding]
        else:
            dx=dXpadded
        self.dx=dx
        self.db=dout.sum(axis=3).sum(axis=2).sum(axis=0)

        #  now dw
        # https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
        # https://github.com/neuron-whisperer/cnn-numpy/blob/master/cnn_numpy.py
        # We can notice that dw is a convolution of the input x with a filter dy
        # ok..so similar to dx.. but what about the shape? dy is the filter? rotate dout twice?
        # no .. i dont
        # may be pad?
        # padding_=(H*self.stride-self.stride+self.kernel_size-H)//2
        # this is incorrect - failing python train.py
        s=self.stride
        k=len(self.weight[0][0])
        f=len(dout[0][0])
        n=H
        # always pad with 1?
        # padding_=(s*k-s+f-n)//2
        padding_=(s*k-s+f-n)//2
        # padding_=(H*self.stride-self.stride+len(dout[image_number])-self.kernel_size)//2
        
        channel_derivatives=[]
        # some combinations are not running
        filter_channel_derivatives=[]
        filter_derivatives=[]
        w_sum=np.zeros([dout.shape[1],C,self.kernel_size,self.kernel_size])
        for filter_number in range(dout.shape[1]):
            filter_derivative=[]
            for image_number in range(N):    
                conv=Conv2D(in_channels=1,
                                    out_channels=1,
                                    kernel_size=len(dout[0][0]),
                                    stride=self.stride,
                                    padding=padding_,new_seed=False)
                conv.weight = np.expand_dims(np.expand_dims(dout[image_number][filter_number],0),0)
                            # no correction this time
                # conv.bias = np.zeros(len(dout))
                channel_derivatives=[]
                for channel_number in range(C):
                    img=np.expand_dims(np.expand_dims(x[image_number][channel_number],0),0)
                    channel_derivative=conv.forward(img)
                    channel_derivative=channel_derivative.reshape(self.weight[0][0].shape)
                    channel_derivatives.append(channel_derivative)
                filter_derivative.append(channel_derivatives)    
                # w_sum[filter_number][channel_number]=np.array(filter_derivative).sum(axis=1)
                
                # w_sum[filter_number][channel_number]=np.array(filter_derivative).sum(axis=0)
            # filter_derivatives.append(filter_derivative)
            w_sum[filter_number]=np.array(filter_derivative).sum(axis=0)
            filter_derivatives=[]

        self.dw=w_sum