import pandas as pd
import torch
import os
import numpy as np
from skimage import io, transform

from utils import *
from fashion_gen_vae import *
from style_transfer import apply_style

def pass_image(model, img):
    """Helper method that passes input image through the model
    no use of resnet here as it falsifies colours a bit in evaluation mode and slows down computation time.
    """
    with torch.no_grad():
        x, z, _, _, _ = model(img)
    return x, z

def decode_latent(model, latent):
    """decodes a latent  vector
    no use of resnet here as it falsifies colours a bit in evaluation mode and slows down computation time.
    """
    x = model.decode(latent)[1].cpu()
    return x

def image_to_tensor(image):
    """Helper method that convers input image to a tensor that can be used by the model
    """
    img = io.imread(image)
    img = transform.resize(img,(128,128))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img = img.unsqueeze(dim=0).to('cuda')
    img = img.clone().detach().float()
    #img = torch.tensor(img, dtype=torch.float32).to('cuda')

    return img

def apply_style_interpolation(model, patterns_model, style_transfer_list, content_image_latent, title='styleTransfer_interpolation', styleTransfer=True):
    """Applies colour-based style transfer to a list of style images for interpolation
    """
    styled_list = []
    i=0
    for style_image_latent in style_transfer_list:
        i+=1
        image = apply_style(content_image_latent, style_image_latent, model, patterns_model, i, title, styleTransfer)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = image.transpose((2, 0, 1))
        img = torch.from_numpy(img) / 255
        img = img.unsqueeze(0).to('cuda')
        img = torch.tensor(img, dtype=torch.float32).to('cuda')
        styled_list.append(img)

    tensors = torch.stack(styled_list, dim=0).squeeze(1).cpu()
    return tensors

def sample_generic(model):
    """Realizes the creation of content vectors. Can either generate completely random images by passing
        a vector of size 1x30x16x16 of the gaussian distribution directly to the decoder, or can add
        attributes to the generation. 
        ----------
        model: variational autoencoder model
        attributes: possible attribute to constrain the image generation
    """
    # sample completely random image
    print('\n\n{}\n__________'.format('Random Sample'))
    z = torch.randn((1, 30, 16, 16)).to('cuda') 
    x = decode_latent(model, z)
    return x, z

def reconstruct(model, image):
    """Reconstructs a given input image.
        ----------
        model: variational autoencoder model
        resnet: resnet moddel
        image: original input image path from the dataset
    """
    img_name = os.path.join('./data/apply_data_shapes/', image)
    img = image_to_tensor(img_name)
    x, z = pass_image(model, img)
    print("reconstruction\n_______________________")
    return x, z

def generate_attribute_vector(model, gender='Mens', attribute=None, attribute2=None):
    """Encodes original images based on the chosen attributes and computes the mean of the latent vector.
    This mean is then returned as attribute vector. Attribute vector is computed by chosing 100 images that
    fullfill the attribut constraints and 100 who does not to compute the difference between them as
    attribute vector.
        ----------
        model: variational autoencoder model
        resnet: resnet moddel
        attributes: chosen attribute constraints
    """
    df = pd.read_csv('./data/preprocessed_shapes_metadata.csv')
    counta = 0
    countb = 0
    
   
    
    att_tensors = []
    neg_att_tensors = []
    #TODO: Remove gender here?
    if attribute is not None and attribute2 is None:
        df_true = df[(df[attribute] == 1) & (df.productGender == '"'+gender+'"')].sample(frac=1)
        df_false = df[(df[attribute] == 0) & (df.productGender == '"'+gender+'"')].sample(frac=1)
        print("\n\nAPPLIED ATTRIBUTES: " + attribute +"\n_________________________________")
        
    elif attribute2 is not None and attribute is None:
        df = df[['itemN','viewId','productGender','ButtonUpShirts','Buttondown','Vneck','Blouses','Workout','Polos','Athletic']]
        df['sum'] = df[['ButtonUpShirts','Buttondown','Vneck','Blouses','Workout','Polos','Athletic']].sum(axis=1)
        df_true = df[(df[attribute2] == 1) & (df['sum'] == 1) & (df.productGender == '"'+gender+'"')].sample(frac=1)
        df_false = df[(df[attribute2] == 0) & (df.productGender != '"'+gender+'"')].sample(frac=1)
        print("\n\nAPPLIED ATTRIBUTES: " + attribute2+"\n_________________________________")
        
    elif attribute is not None and attribute2 is not None:
        df_true = df[(df[attribute] == 1) & (df[attribute2] == 1) & (df.productGender == '"'+gender+'"')].sample(frac=1)
        df_false = df[(df[attribute] == 0) & (df[attribute2] == 0) ].sample(frac=1)
        print("\n\nAPPLIED ATTRIBUTES: " + attribute +" and " +attribute2+"\n_________________________________")
        
    else:
        return None
    
    if len(df_true) < 5:
        print("LEN smaller 5")
        return None
        
    print(df_true.head(3))
    print(df_false.head(3))
    
    print(str(len(df_true)) + " matching garment shapes found.")
    for row in df_true.iterrows():
        img_name = os.path.join('./data/preprocessed_data_shapes/' + str(row[1]['itemN'])+'_'+str(row[1]['viewId'])+'.jpg')
        img = image_to_tensor(img_name)  
        x, enc = pass_image(model, img)
        counta += 1
        att_tensors.append(enc)
            
        if counta >= 50:
            break
                
    for row in df_false.iterrows():
        img_name = os.path.join('./data/preprocessed_data_shapes/' + str(row[1]['itemN'])+'_'+str(row[1]['viewId'])+'.jpg')
        img = image_to_tensor(img_name)  
        x, enc = pass_image(model, img)
        countb += 1
        neg_att_tensors.append(enc)

        if countb >= 50:
            break         
    
    #compute attribute by computing mean and then the difference between attribute and non-attribute latents
    z_att = torch.stack(att_tensors, dim=0)
    z_att = torch.mean(z_att, dim=0)
    z_neg = torch.stack(neg_att_tensors, dim=0)
    z_neg = torch.mean(z_neg, dim=0)

    att_final = z_att - z_neg
    return att_final

def apply_attribute(model, z, att, num_steps=8, max_att=3, show=False):
    """Applies weighted attribute vector to a content vector by simple addition. Weight increased
    through linear annealing.
        ----------
        model: variational autoencoder model
        resnet: resnet moddel
        z: latent vector to apply the attribute vectors to
        att: attribute vector to be applied
        num_steps: number of steps to change weight of attribute vector from 0 to max_att
        max_att: maximum weight of attribute vector
    """
    print("apply_attribute")
    attributes_applied = []
    decoded_images = []
    for i in range(num_steps):
        attributes_applied.append(z + linear_annealing(start=0, final=max_att, step=i, steps=num_steps) * att)

    for j, latent_vector in enumerate(attributes_applied):
        decoded_images.append(decode_latent(model, latent_vector))
        if show is True:
            show_images(decoded_images[j],'attribute_application_step_{}'.format(j))

    images = torch.stack(decoded_images).squeeze(dim=1)
    return images, attributes_applied


def decode_pattern(model, pattern_image):
    """Simply decodes pattern
        ----------
        model: variational autoencoder model
        resnet: resnet moddel
        pattern_image: pattern image
    """
    img_name = os.path.join('./data/apply_data_styles/', pattern_image)
    img = image_to_tensor(img_name)

    x, z = pass_image(model, img)
    return x, z
