import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as T
import PIL
from image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
import matplotlib.pyplot as plt

def preprocess(img, size=512):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def features_from_img(imgpath, imgsize, cnn, dtype):
    img = preprocess(PIL.Image.open(imgpath), size=imgsize)
    img_var = Variable(img.type(dtype))
    return extract_features(img_var, cnn), img_var

# Older versions of scipy.misc.imresize yield different results
# from newer versions, so we check to make sure scipy is up to date.
def check_scipy():
    import scipy
    vnums = list(map(int, scipy.__version__.split('.')))
    assert vnums[1] >= 16 or vnums[0] >= 1, "You must install SciPy >= 0.16.0 to complete this notebook."


# We provide this helper code which takes an image, a model (cnn), and returns a list of
# feature maps, one per layer.
def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.

    Inputs:
    - x: A PyTorch Variable of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.

    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Variable of shape (N, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features


def style_transfer(name, content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, content_loss, style_loss, tv_loss, cnn, dtype, init_random=False):
    """
    Run style transfer!

    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """

    # Extract features for the content image
    content_img = preprocess(PIL.Image.open(content_image), size=image_size)
    content_img_var = Variable(content_img.type(dtype))
    feats = extract_features(content_img_var, cnn)
    content_target = feats[content_layer].clone()

    # Extract features for the style image
    style_img = preprocess(PIL.Image.open(style_image), size=style_size)
    style_img_var = Variable(style_img.type(dtype))
    feats = extract_features(style_img_var, cnn)
    style_targets = []
    for idx in style_layers:
        style_targets.append(style_loss.gram_matrix(feats[idx].clone()))

    # Initialize output image to content image or nois
    if init_random:
        img = torch.Tensor(content_img.size()).uniform_(0, 1)
    else:
        img = content_img.clone().type(dtype)

    # We do want the gradient computed on our image!
    img_var = Variable(img, requires_grad=True)

    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180

    # Note that we are optimizing the pixel values of the image by passing
    # in the img_var Torch variable, whose requires_grad flag is set to True
    optimizer = torch.optim.Adam([img_var], lr=initial_lr)

    f, axarr = plt.subplots(1, 2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(deprocess(content_img.cpu()))
    axarr[1].imshow(deprocess(style_img.cpu()))
    plt.savefig('styles_images/' + name + '_before.png')
    plt.show()
    plt.figure()

    for t in range(200):
        if t < 190:
            img.clamp_(-1.5, 1.5)
        # logic for scheduler
        # i think even the location of optimizer is causing grains on the image
        optimizer.zero_grad()
        #    there is too much grain

           
        feats = extract_features(img_var, cnn)
        # not sure if this is the issue
        # i really hate this garbage step.. 
        # this has to be the most useless line of code every made
        # first and second image looks same?
        # no it is different.. the knot moved.. so something did happen
        # optimizer.zero_grad()
        #### Add to-do to have the students implement the back-propagation part

        ##############################################################################
        # TODO: Implement this update rule with by forwarding it to criterion        #
        # functions and perform the backward update.                                 #
        #                                                                            #
        # HINTS: all the weights, loss functions are defined. You don't need to add  #
        # any other extra weights for the three loss terms.                          #
        # The optimizer needs to clear its grad before backward in every step.       #
        #                                                                            #
        # NOTE: There is a final optimization needed to get good style transferred   #
        #   images. Do look at the variables 'decay_lr_at' and 'decayed_lr'.         #
        #   You would need to reduce the learning rate for the last few epochs.      #
        ##############################################################################

        #     The update rule in function
        # style_transfer of style_utils.py is held out for you to finish        
        # okay.. what is coming here.. lol .. so many parameters
        # looks like a scheduler is required for reducing learning rates for last epoch
        # where is the loop? 200 iterations
        # only backprop is required.. need to think.. what is ahppening
        # most likely need to call the two losses and add them
        # what are the dimension of losses? are they a list?
        # everything is list
        # just use the test cases?
        # how will this work? 
        # content loss
        # then style loss.. 
        # add them together?
        # then? loss.backward()
        # update current image?
        # what about tv loss? may be add that as well to style loss and content loss
        # feats is overwritten multiple times. this is most likely wrong
        # c_feats is first.. order is wrong
        # but order wont matter.. it is sum of squares of difference. so issue is somewhere else
        content_l = content_loss(content_weight,feats[content_layer],content_target)
        # # i did not calculate c_feats
        # # unfortunately.. style want all the feats
        style_l = style_loss(feats, style_layers, style_targets, style_weights)
        # should this be content or img? 
        # content is fixed.. img_var is updated every loop
        # penalty should be on input image
        tv_l = tv_loss(img_var, tv_weight)
        # why is the content loss zero? because everything is content target
        # content_loss(content_weight, c_feats[content_layer], feats[content_layer]).data.numpy()
        loss=content_l+style_l+tv_l
        # print(img_var.sum())
        loss.backward()
        # update x? which x?
        # how are we calling this function? it should contain the name of the file
        # style_transfer(**params1)
        # this is deflating dictionary
        # params1 = {
        #     'name': 'composition_vii_tubingen',
        #     'content_image' : 'styles_images/tubingen.jpg',
        #     'style_image' : 'styles_images/composition_vii.jpg',
        #     'image_size' : 192,
        #     'style_size' : 512,
        #     'content_layer' : 3,
        #     'content_weight' : 5e-2,
        #     'style_layers' : (1, 4, 6, 7),
        #     'style_weights' : (20000, 500, 12, 1),
        #     'tv_weight' : 5e-2,
        #     'content_loss' : content_loss,
        #     'style_loss': style_loss,
        #     'tv_loss': tv_loss,
        #     'cnn': cnn,
        #     'dtype': dtype
        # }
        # i think name is the name of the starting file 
        # it is not used anywhere in the code
        # how is it possible.. it should be used for calculating the two losses-
        # put a trace.. bad coding style.. feats is overwritten multiple times
        # name
        # 'composition_vii_tubingen'
        # content_image
        # 'styles_images/tubingen.jpg'
        # style_image
        # 'styles_images/composition_vii.jpg'
        # okay.. i think img is the random image and we save it to name
        # should this be minus or plus?  this is gradient descent
        # learning rate?
        # i already have a dedicated optimizer
        # print(img_var.sum())
        if t==decay_lr_at:
            # https://stackoverflow.com/questions/48324152/pytorch-how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no
           optimizer.param_groups[0]['lr'] = decayed_lr
        # if i put optimizer.zero_grad() here, it will not initiate style transfer at all   
        optimizer.step()
        # print(img_var.sum())
        # do i need to update the image?
        # optimizer.step should update the variable as well
        # it changes
        # check=1
        # but.. this should not update the weights.. and only the dx part
        # let me think.. do we have weights? no.. actually.. there is no linear part
        # we directly optimize the image.. so this should work
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

        # if t % 100 == 0:
        #     print('Iteration {}'.format(t))
        #     plt.axis('off')
        #     plt.imshow(deprocess(img.cpu()))
        #     plt.show()
    plt.axis('off')
    plt.imshow(deprocess(img.cpu()))
    plt.savefig('styles_images/' + name + '.png')
    plt.show()