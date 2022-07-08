import zipfile
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torchvision
from torch.nn.utils import spectral_norm
import imageio
import numpy as np
import torchvision.models as models
import cv2
from skimage import io
from random import randrange


from fashion_gen_vae import MY_VAE


""" Pretrained vgg for style transfer and perceptual loss"""
vgg = models.vgg19(pretrained=True).to('cuda')

for param in vgg.parameters():
    param.requires_grad = False

vgg.eval()

def load_current_models(check_list = ['weights_shapeVAE.tar','weights_styleVAE.tar']):
    """ Loads the current models"""
    mvc_model = MY_VAE(batch_size=16, img_size=128).cuda()
    mvc_model.apply(add_sn)
    cp, _, _ = load_model(check_list[0])
    mvc_model.load_state_dict(cp['model_state_dict'])
    for param in mvc_model.parameters():
        param.requires_grad = False
    print("shape model loaded")

    patterns_model = MY_VAE(batch_size=16, img_size=128).cuda()
    patterns_model.apply(add_sn)
    cp, _, _ = load_model(check_list[1])
    patterns_model.load_state_dict(cp['model_state_dict'])
    for param in patterns_model.parameters():
        param.requires_grad = False
    print("models loaded successfully")
    return mvc_model, patterns_model

def unzip_datasets():
    """ unzips the datasets"""
    with zipfile.ZipFile('./data/apply_data_styles.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')
    with zipfile.ZipFile('./data/apply_data_shapes.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')
    with zipfile.ZipFile('./data/preprocessed_data_shapes.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')

def save_creations(img, name, epoch= None):
    """ Creates a grid of images that are passed to the method as tensors.
     Saves the image in the output folder given the name and eventually the epoch."""
    images = torchvision.utils.make_grid(img)
    plt.imshow(images.permute(1,2,0))
    if epoch is not None:
        plt.savefig("./output/{}_epoch_{}.png".format(name,epoch))
    else:
        print("saved")
        plt.savefig("./output/{}.png".format(name))


def load_model(checkpoint):
    """ Loads the state dict, the epoch and the loss of a model to either continue training or
        to evaluate/use the model.
        ----------
        checkpoint: path to the checkpoint to load
    """
    cp = torch.load('./data/'+checkpoint)
    epoch = cp['epoch']
    loss = cp['loss']
    return cp, epoch, loss

def show_images(images, name):
    """ Plots the given images in a grid to console.
        ----------
        images: (possibly stacked) image tensors
    """
    images = torchvision.utils.make_grid(images.cpu())
    plt.title(name)
    images = images.cpu().detach().numpy().transpose(1, 2, 0) * 255
    cv2.imwrite('./output/{}.jpg'.format(name),cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

def init_weights(m):
    """ Initializes the weights based on a xavier uniform distribution.
        ----------
        images: (possibly stacked) image tensors
    """
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform(m.weight)
        if type(m) != torch.nn.Linear:
            m.bias.data.fill_(0.01)

def add_sn(m):
    """ Weight normalization technique that leads to better training stability according to Vahdat & Kanutz (2020)
        ----------
        m: model

        Implementation References
        --------------
        Taken from https://github.com/GlassyWing/nvae
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        return spectral_norm(m)
    else:
        return m

def linear_annealing(start, final, step, steps):
    """ Method that represents a linear function. Is used for training annealing schedule and linear interpolation.
        ----------
        start: starting value
        final: final value
        step: current step out of the number of steps
        steps: number of steps to get from start value to final value
    """
    return start + ((final-start)*(step))/(steps-1)


def slerp(val, low, high):
    """https://github.com/soumith/dcgan.torch/issues/14"""
    low = low.numpy()
    high = high.numpy()
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high



def interpolation(z1, z2, num_steps, model, title=None):
    """ Realizes linear interpolation between two latent vectors.
            ----------
            z1: beginning vector
            final: final vector
            num_steps: number of steps to complete the interpolation from z1 to z2
            model, resnet: corresponding models
    """

    steps = np.linspace(0, 1, num=num_steps)
    interpolated = []
    slerp_list = []
    slerp_z_list = []
    interpolatedz = []
    i = 0
    for step in steps:
        #computes intermediate values between z1 and z2, adds them to a latent list and an image list
        interpolatedz.append((1.0 - step) * z1 + step * z2)
        slerp_val = slerp(linear_annealing(0, 1, i, num_steps), z1.reshape(-1).cpu(), z2.reshape(-1).cpu())
        slerp_z_list.append(torch.tensor(slerp_val, dtype=torch.float32).view(1,30,16,16).cuda())

        slerp_list.append(model.decoder(torch.tensor(slerp_val, dtype=torch.float32).view(1,30,16,16).to('cuda'))[1])
        interpolated.append(model.decoder(((1.0 - step) * z1 + step * z2))[1])
        
        img = slerp_list[i].squeeze(0).cpu().detach().numpy().transpose(1, 2, 0) * 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./output/{}_step_{}.jpg'.format(title,i),img)
        i+=1

    show_images(torch.stack(interpolated, dim=0).squeeze(1).cpu(), "traditional_"+title)
    show_images(torch.stack(slerp_list, dim=0).squeeze(1).cpu(), "slerp_"+title)
    return interpolated, interpolatedz, slerp_list, slerp_z_list

def overlay_styles(style1, style2, title):
    style1 = style1[0].cpu().detach().numpy().transpose(1, 2, 0) * 255
    style1 = cv2.cvtColor(style1, cv2.COLOR_BGR2RGB)
    
    style2 = style2.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0) * 255
    style2 = cv2.cvtColor(style2, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(style1, 0.5, style2, 0.5, 1)
    cv2.imwrite('./output/overlay_image_{}.jpg'.format(title),overlay)
    
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    overlay = overlay.transpose((2, 0, 1))
    overlay = torch.from_numpy(overlay) / 255
    overlay = overlay.unsqueeze(0).to('cuda')
    overlay = torch.tensor(overlay, dtype=torch.float32).to('cuda')
          
    return overlay


def plot_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.show()
