import torch
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt

from image_utils import preprocess
class SaliencyMap:
    def compute_saliency_maps(self, X, y, model):
        """
        Compute a class saliency map using the model for images X and labels y.

        Input:
        - X: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the saliency map.

        Returns:
        - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
        images.
        """
        # Make sure the model is in "test" mode
        model.eval()

        # Wrap the input tensors in Variables
        X_var = Variable(X, requires_grad=True)
        y_var = Variable(y, requires_grad=False)
        saliency = None

        ##############################################################################
        # TODO: Implement this function. Perform a forward and backward pass through #
        # the model to compute the gradient of the correct class score with respect  #
        # to each input image.                                                       #
        #                                                                            #
        ##############################################################################
        # It only needs a single back-propagation pass
        # set the optimization parameters
        # pass
        # is dx ? it should be.. so it need hooks? i think we are using Variable function
        # run the forward pass
        # maybe this X is wrong
        output=model.forward(X_var)
        #  To compute it, we compute thegradient of the unnormalized score 
        # corresponding to the correct class (whichis a scalar) with respect to the pixels of the image

        # gather exact coordinates we want based on the y target
        # so saliency map for all images - 5 images
        # the correct value of y is the loss function 
        # tensor([958,  85, 244, 182, 294]) this is y
        loss_individual=output.gather(1,y_var.reshape(len(y),1))
        loss=loss_individual.sum()
        # perform backward pass
        loss.backward()
        # get the gradients
        X_var_grad=X_var.grad.data
        # compute absolute value
        X_var_grad=torch.abs(X_var_grad)
        # max across dimension channel
        # somehow, I am not getting the shape
        # https://stackoverflow.com/questions/60847083/attributeerror-torch-return-types-max-object-has-no-attribute-dim-maxpool
        # if you specify other args (for example dim=0),
        #  max() function will returns a namedtuple: (values, indices). I guess the values is what you want.
        X_var_grad_saliency=torch.max(X_var_grad,axis=1)
        # what is X.shape?([5, 3, 224, 224])
        # so it should be [5,224,224]

        saliency=X_var_grad_saliency[0]
        # the backward has to be on the raw scores and not on the loss function

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return saliency

    def show_saliency_maps(self, X, y, class_names, model):
        # Convert X and y from numpy arrays to Torch Tensors
        X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
        y_tensor = torch.LongTensor(y)

        # Compute saliency maps for images in X
        saliency = self.compute_saliency_maps(X_tensor, y_tensor, model)
        # Convert the saliency map from Torch Tensor to numpy array and show images
        # and saliency maps together.
        saliency = saliency.numpy()

        N = X.shape[0]
        for i in range(N):
            plt.subplot(2, N, i + 1)
            plt.imshow(X[i])
            plt.axis('off')
            plt.title(class_names[y[i]])
            plt.subplot(2, N, N + i + 1)
            plt.imshow(saliency[i], cmap=plt.cm.gray)
            plt.axis('off')
            plt.gcf().set_size_inches(12, 5)
        plt.savefig('visualization/saliency_map.png')
        plt.show()
