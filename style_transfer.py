from __future__ import print_function
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import torch.optim as optim
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
from utils import plot_image, vgg

"""this file contains the predefined skin detector class and the style transfer model. Only small changes,
as change of the optimizer and the weights; 
Style transfer taken from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
https://github.com/pytorch/tutorials"""

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# This type of optimizer was prefered by the author of the paper
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.SGD([input_img.requires_grad_()], lr=0.00005, momentum=0.95)
    #optimizer = optim.LBFGS([input_img.requires_grad_()], lr=0.01)
    #optimizer = optim.AdamW([input_img.requires_grad_()], lr=0.002)
    return optimizer


# Custom content loss
# \]|c>:/']'0 Vavi's code
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# This is for the style loss
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

# Same structure as the content loss
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to('cuda')

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=10000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 100 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction to have the tensors between 0 and 1
    input_img.data.clamp_(0, 1)

    return input_img

def apply_style(content_latent, style_latent, mvc_model, patt_model, num_iter=0):
    """Method that prepares the images for the colour-based style transfer.
    First, latent vectors are decoded and converted to numpy arrays.
    Second, the predifined skin-detector class is called to segment the skin and create a mask.
    Third, perform_style transfer is called
         ----------
        content_latent: latent vector of content image
        style_latent: latent vector of style image
    """
    #using the pre-implemented skin detector to generate the skin-mask
    content_image_tensor = mvc_model.decode(content_latent)[1]
    style_image_tensor = patt_model.decode(style_latent)[1]

    content_image = cv2.normalize(content_image_tensor.squeeze(0).permute(1,2,0).cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)

    style_image = cv2.normalize(style_image_tensor.squeeze(0).permute(1,2,0).cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)

    skin_seg, back_seg = color_segmentation(content_image)
    
    #converting the skin-mask-images back to RGB
    #color_seg = cv2.cvtColor(color_seg, cv2.COLOR_BGR2RGB)

    stm1 = style_transfer_model()
    result = stm1.perform_style_transfer(content_image, content_image_tensor, style_image, style_image_tensor, skin_seg, back_seg, num_iter)
    return result

class style_transfer_model():
    """Class that defines the styl transfer model, initialized with deeplabv3 for segmentation
        ----------
        content_latent: latent vector of content image
        style_latent: latent vector of style image

        Implementation references
        ------------
        Main idea for foreground-background-segmentation taken from https://www.learnopencv.com/applications-of-foreground-background-separation-with-semantic-segmentation/
        Implemented the details and the mapping of different masks onto the different images
    """
    
    def perform_style_transfer(self, content_image, content_img_tensor, style_image, style_img_tensor, skin_seg ,back_seg, num_iter):
      
        kernel = np.ones((3,3), np.uint8)
        skin_seg = cv2.morphologyEx(skin_seg, cv2.MORPH_OPEN, kernel)
        
        skin_seg = cv2.morphologyEx(skin_seg, cv2.MORPH_CLOSE, kernel)        

        skin_seg = cv2.GaussianBlur(skin_seg, (1,1),0)
        
        
        input_img = content_img_tensor.clone()
        
        # VGG network is normalized with special values for the mean and std
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).cuda()
        
        #apply the style transfer
        output = run_style_transfer(vgg.features.cuda().eval(), cnn_normalization_mean, cnn_normalization_std,
                                    content_img_tensor, style_img_tensor, input_img, num_steps=20000)
        
        
        #convert tensor into numpy uint array
        foreground_img = output.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0) * 255
        foreground_img = foreground_img.astype(np.uint8)
        foreground_img = cv2.cvtColor(foreground_img, cv2.COLOR_BGR2RGB)
        
        #add the styled image to the clothing mask image such that only the clothing is styled 
        back_seg = cv2.cvtColor(back_seg,cv2.COLOR_GRAY2RGB)
        back_seg = cv2.bitwise_not(back_seg)
        back_seg = cv2.GaussianBlur(back_seg, (1,1),0)
        back_seg = cv2.bitwise_not(back_seg)
        back_seg[0:35,0:128] = cv2.add(back_seg[0:35, 0:128], content_image[0:35, 0:128])
        back_seg[122:128,0:128] = cv2.add(back_seg[122:128, 0:128], content_image[122:128, 0:128])
        
        clothing_mask = back_seg < 0.8

        back_seg[clothing_mask] = foreground_img[clothing_mask]
        alpha_mask = skin_seg > 0.5 
        back_seg[alpha_mask] = content_image[alpha_mask]
        
        
        cv2.imwrite('./output/styleTransfer_interpolation_{}.jpg'.format(num_iter),back_seg)

       

        return back_seg

def color_segmentation(image):      
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binary_mask_image = hsv_image
        
    #background
    lower_HSV_values = np.array([0,0,220], dtype="uint8")
    upper_HSV_values = np.array([179,25,255], dtype="uint8")
       
    #skin
    lower_HSV_values4 = np.array([0,0,0], dtype="uint8")
    upper_HSV_values4 = np.array([30,180,235], dtype="uint8")
        
    # A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
    back_seg = cv2.inRange(hsv_image, lower_HSV_values, upper_HSV_values)
    skin_seg = cv2.inRange(hsv_image, lower_HSV_values4, upper_HSV_values4)

    return skin_seg, back_seg