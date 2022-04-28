from __future__ import print_function, division
from collections import  OrderedDict
import time
from torch.nn import functional as F
import torch
import random
import os

from torch.utils.data import *
from fashion_gen_vae import *
from utils import *
from dataloader import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



torch.autograd.set_detect_anomaly(False)

class FLPLoss():
    """ FLPLoss class to compute perceptual loss by using the features extracted from a vgg.
    As described by Hou et al. (2019)
            ----------
            x_features: features of input image
            recon_x_features: features of recon image
            x: original image
            recon_x: reconstructed image
    """
    def compute_loss(self, x_features, recon_x_features, x, recon_x, reduction='mean'):

        feature_loss = 0.0
        for r, i in zip(recon_x_features, x_features):
            feature_loss += F.mse_loss(r, i)

        feature_loss 

        return feature_loss

def split_dataset(dataset, data_type='Shape'):
    """ Splits dataset 70/10/20 in training and test set, if wanted
    """
    if data_type=='Shape':
        lengths = [int(len(dataset)*0.7)+2, int(len(dataset)*0.1), int(len(dataset)*0.2)]
    else:
        lengths = [int(len(dataset)*0.7), int(len(dataset)*0.1), int(len(dataset)*0.2)]
    train_data, val_data, test_data  = random_split(dataset, lengths)
    print("len_train_data: ", len(train_data))
    print("len_test_data: ", len(test_data))
    print("len_val_data: ", len(val_data))
    return train_data, val_data, test_data

def extract_features(vgg_model, x, model='vae1234'):
    """ Extracts features of different layers when passing data x through pretrained vgg net.
        ----------------
        vgg_model: pretrained vgg-net
        x: data

        Implementation references
        ----------------
        taken from https://github.com/AntixK/PyTorch-VAE/blob/master/models/dfcvae.py
        Slightly changed and added additional layers to compute the loss
    """
    if model == 'vae123':
        feature_layers = [1, 6, 11]
    elif model == 'vae345':
        feature_layers = [11, 20, 29]
    else:
        feature_layers = [1, 6, 11, 20, 29, 34]

    features = []
    result = x
    for (key, module) in vgg_model.features._modules.items():
        result = module(result)
        if feature_layers.__contains__(int(key)):
            features.append(result)

    return features

def init_weights(m):
    """ Weight initialization to xavier uniform distribution
    """
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform(m.weight)

class Storer():
    """ Storer class that saves given values in a csv document after each epoch.
        """
    def __init__(self):
        super().__init__()
        self.run_data = []
        self.path = None
        self.start_time = time.time()

    def set_values(self, epoch, elbo, recon, flp=-1, kld=-1, resnet=-1, nan_count=0):
        results = OrderedDict()
        results['epoch'] = epoch
        results['elbo'] = elbo
        results['recon_loss'] = recon
        results['flp_loss'] = flp
        results['kld_loss'] = kld
        results['run_duration'] = time.time() - self.start_time
        results['val_nan_count'] = nan_count

        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        df.to_csv(self.path)

    def set_path(self, path, val = False):
        if val == True:
            self.path = './' + path +'_'+str(random.randint(0,100))+"val_results.csv"
        else:
            self.path = './' + path +'_'+str(random.randint(0,100))+"results.csv"

def shape_linear_annealing_loss(iteration, epoch, kld_loss, length, batch_size, factor=1):
    """ Self-created cyclic annealing schedule as described in the thesis; uses linear annealing
    with different values to increase the weight to the kld loss in a cyclic way.
    Increases linear anneling values after some epochs.
    num_steps computed for the iterations in each epoch; iteration computed to add num_steps * epoch to
    the linear-annealing step value; Linear annealing starts again every 3 epochs
    ------------------------
    iteration: current iteration in the epoch
    epoch: current epoch
    length: length of train dataset
    """
    num_steps = int(length / batch_size) + 1
    iteration = iteration + ((epoch-1)%2) *num_steps 
    annealing = 0
    x = factor 
    if epoch <= 2:
        annealing = linear_annealing(start=0.05, final=0.2, step=iteration, steps=num_steps*2) * x  
    elif epoch <= 4:
        annealing = linear_annealing(start=0.0001, final=0.005, step=iteration, steps=num_steps*2) * x  
    elif epoch <= 6:
        annealing = linear_annealing(start=0.0005, final=0.01, step=iteration, steps=num_steps*2) * x  
    elif epoch <= 8:
        annealing = linear_annealing(start=0.001, final=0.1, step=iteration, steps=num_steps*2) * x  
    elif epoch <= 10:
        annealing = linear_annealing(start=0.01, final=0.2, step=iteration, steps=num_steps*2) * x  
    elif epoch <= 18:
        annealing = linear_annealing(start=0.05, final=0.5, step=iteration, steps=num_steps*4) * x  
    elif epoch <= 26:
        annealing = linear_annealing(start=0.1, final=1, step=iteration, steps=num_steps*4) * x   
    elif epoch <= 34:
        annealing = linear_annealing(start=0.5, final=1.5, step=iteration, steps=num_steps*4) * x  
    elif epoch <= 42:
        annealing = linear_annealing(start=1, final=3, step=iteration, steps=num_steps*4) * x 
    elif epoch <= 50:
        annealing = linear_annealing(start=2, final=6, step=iteration, steps=num_steps*4) * x 
    elif epoch <= 60:
        annealing = 5 * x
    elif epoch <= 70:
        annealing = linear_annealing(start=5, final=7, step=iteration, steps=num_steps*5) *x
    elif epoch <= 80:
        annealing = 7 * x
    elif epoch <= 90:
        annealing = linear_annealing(start=5, final=10, step=iteration, steps=num_steps*5) *x
    elif epoch <= 100:
        annealing = 10*x
    elif epoch <= 110:
        annealing =  linear_annealing(start=10, final=20, step=iteration, steps=num_steps*10) *x
    else:
        annealing = 20*x
        
    if (iteration % 100 == 0):
        print("Iteration: ",iteration," - Annealing_value: ",annealing)
    return annealing * kld_loss

def style_linear_annealing_loss(iteration, epoch, kld_loss, length, batch_size, factor=1):
    num_steps = int(length / batch_size) + 1
    iteration = iteration + ((epoch-1)%2) *num_steps 
    annealing = 0
    x = factor 
    if epoch <= 2:
        annealing = linear_annealing(start=0.05, final=0.2, step=iteration, steps=num_steps*2) * x  
    elif epoch <= 10:
        annealing = linear_annealing(start=0.0001, final=0.005, step=iteration, steps=num_steps*4) * x  
    elif epoch <= 18:
        annealing = linear_annealing(start=0.0005, final=0.01, step=iteration, steps=num_steps*4) * x  
    elif epoch <= 26:
        annealing = linear_annealing(start=0.001, final=0.1, step=iteration, steps=num_steps*4) * x  
    elif epoch <= 34:
        annealing = linear_annealing(start=0.01, final=0.5, step=iteration, steps=num_steps*4) * x
    elif epoch <= 42:
        annealing = linear_annealing(start=0.05, final=0.5, step=iteration, steps=num_steps*4) * x  
    elif epoch <= 50:
        annealing = linear_annealing(start=0.1, final=1, step=iteration, steps=num_steps*4) * x   
    else:
        annealing = 1*x
    
    if (iteration % 100 == 0):
        print("Iteration: ",iteration," - Annealing_value: ",annealing)
    return annealing * kld_loss

def show_progress(images, image_recon, z, model, epoch, path, save = False):
    """ Method that can print outputs to the console after some iterations.
    """
    with torch.no_grad():
        interpolation(z[0].unsqueeze(0), z[2].unsqueeze(0), num_steps=32, model=model, title='interpolation_{}'.format(epoch))
        if save == True:
            z2 = torch.randn((16, 30, 16 ,16)).cuda()
            recon = model.decode(z2)[1]
            save_creations(images.cpu(), "images_origin_", epoch)
            save_creations(image_recon.cpu(), "images_recon", epoch)
            save_creations(recon.cpu(),"samples_", epoch)
            
def test_model(model, testLoader, dataset):
    total_test_loss = 0
    total_test_recon_loss = 0
    total_test_kld_loss = 0
    total_test_flp_loss = 0

    torch.cuda.empty_cache()
    for j, test_batch in enumerate(testLoader):
        #iterate batchwise through testLoader
        flp_loss_function = FLPLoss()
        with torch.no_grad():

            # extracting the images from the batch
            test_images = test_batch['image'].to('cuda')
            test_images = torch.tensor(test_images, dtype=torch.float32).to('cuda')

            # passing the images through the model and the results through the resnet
            test_image_recon, test_z, mu, log_var, test_kld_loss = model(test_images)

            # calculating the features of the correspondig Conv2d-layers of the vgg-net
            x_features = extract_features(vgg, test_images, model='')
            x_recon_features = extract_features(vgg, test_image_recon, model='')

            # calculating the losses; recon_loss only for convenience reasons
            test_flp_loss = flp_loss_function.compute_loss(x_features, x_recon_features, test_images, test_image_recon)
            test_mse = F.mse_loss(test_image_recon, test_images, reduction='mean')

                          
            test_kl_sum = 0
            for kl_loss in test_kld_loss:
                test_kl_sum += kl_loss
            if dataset == 'Shape':
                test_loss = test_flp_loss+shape_linear_annealing_loss(j, 149, test_kl_sum, len(testLoader), 16, 1)+test_mse
            else:
                test_loss = test_flp_loss+shape_linear_annealing_loss(j, 57, test_kl_sum, len(testLoader), 16, 0.1)+test_mse
            total_test_loss += test_loss.item()
            total_test_kld_loss += test_kl_sum.item()
            total_test_flp_loss += test_flp_loss.item()
            total_test_recon_loss += test_mse.item()
    
    print("Total loss: ", total_test_loss / len(testLoader))
    print("Total kld: ", total_test_kld_loss / len(testLoader))
    print("Total flp: ", total_test_flp_loss / len(testLoader))
    print("Total recon: ", total_test_recon_loss / len(testLoader))  

            
def train_model(dataset, batch_size, img_size, path, opt='SGD', lr=0.01, momentum=0.95, factor=1, checkpoint=None, mode='train'):
    """ Training method. Flow explained with comments in the code
        -------------------
    """
    # loading the models, applying spectral normalizatio as Vahdat & Kanutz (2020, NVAE) proposed
    # Weight initialization
    model = MY_VAE(batch_size=batch_size, img_size=img_size).cuda()
    
    train_storer = Storer()
    val_storer = Storer()

    model.apply(init_weights)
    model.apply(add_sn)

    save = True

    epoch = 1

    flp_loss_function = FLPLoss()

    epochs = 300

    train_storer.set_path(path)
    val_storer.set_path(path, val = True)

    if dataset == 'Shape':
        train_data, val_data, test_data = split_dataset(create_new_data_squared(img_size), dataset)
    elif dataset == 'Style':
        train_data, val_data, test_data = split_dataset(create_newPatt_dataset(img_size), dataset)
    
    trainLoader = DataLoader(train_data, batch_size, shuffle=True, num_workers=8)
    testLoader = DataLoader(test_data, batch_size, shuffle=True, num_workers=8)
    valLoader = DataLoader(val_data, batch_size, shuffle=True, num_workers=8)

    if opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2, eta_min=0.00001, last_epoch=-1)
    elif opt=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=momentum)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=momentum)
        

    # loading a checkpoint if training should be continued from an earlier savepoint
    if checkpoint is not None:
        cp, epoch, loss = load_model(checkpoint)
        model.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        scheduler.load_state_dict(cp['scheduler_state_dict'])

        save = False
        
    if mode == 'test':
        with torch.no_grad():
            test_model(model, testLoader, dataset)
            epoch = 350
            

    while epoch < epochs:
        print("\n EPOCH {}\n________________".format(epoch))
        total_loss = 0
        total_recon = 0
        total_kld = 0
        total_flp = 0
        total_resnet = 0
        total_resnet_flp = 0

        for i, batch in enumerate(trainLoader):
            #iterate batchwise through trainloader
            torch.cuda.empty_cache()
            model.train()

            # extracting the images from the batch
            images = batch['image'].to('cuda')
            images = torch.tensor(images, dtype=torch.float32)
            images = images.to('cuda')

            # passing the images through the model and the results through the resnet
            image_recon, z, mu, log_var, kld_loss = model(images)

            # calculating the features of the correspondig Conv2d-layers of the vgg-net
            x_features = extract_features(vgg, images, model='')
            x_recon_features = extract_features(vgg, image_recon, model='')

            # calculating the losses; recon_loss only for convenience reasons
            flp_loss = flp_loss_function.compute_loss(x_features, x_recon_features, images, image_recon)

            mse = F.mse_loss(image_recon, images, reduction='mean')


            # Adding up the kl-losses and applying the annealing schedule
            kl_sum = 0
            for k, kl_loss in enumerate(kld_loss):
                kl_sum += kl_loss
            if dataset == 'Shape':
                loss = flp_loss + shape_linear_annealing_loss(i, epoch, kl_sum, len(train_data), batch_size, factor) + mse
            else:
                loss = flp_loss + style_linear_annealing_loss(i, epoch, kl_sum, len(train_data), batch_size, factor)  + mse

            #backpropagation with gradient clipping
            optimizer.zero_grad()

            model.requires_grad = True
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()

            if i >= 50 or epoch > 1:
                #add batch losses to total losses of the epoch
                total_loss += sum(kld_loss).item()
                total_flp += flp_loss.item()
                total_kld += kl_sum.item()
                total_recon += mse.item()
            else:
                total_loss += 10
                total_flp += 10
                total_kld += 10
                total_recon += 10

            if dataset == 'Shape' or dataset == 'Style':
                if ((i % 500) == 20 or i == 50 or i == 150 or i == 350):
                    torch.cuda.empty_cache()
                    print("\n--------------------------------\niteration: ", i)
                    print("learning_rate: ",scheduler.get_lr())
                    print("batch_kl_losses: ",kld_loss)
                    print("avg_flp_loss:\t\t ", total_flp / (i + 1))
                    print("avg_recon_loss:\t\t ", total_recon /(i + 1))
                    print("batch_total_loss:\t ", loss.item() / (i + 1))
                    print("average_total_loss:\t ", total_loss / (i + 1))
                    print("average_kld:\t\t ", total_kld / (i + 1))
                    if i == 150 or i == 1020:
                        with torch.no_grad():
                            show_progress(images, image_recon, z, model, epoch, path, save = True)
                    
                    save = True

                if (epoch-1) % 2 == 0 and epoch != 1 and i == 10 and save is not False:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                    }, "./{}vae_{}_{}.tar".format(path, epoch, dataset))
    
        val_total_loss = 0
        val_total_recon = 0
        val_total_kld = 0
        val_total_flp = 0

        torch.cuda.empty_cache()
        count = 0
        nan_count = 0
        for j, val_batch in enumerate(valLoader):
            #iterate batchwise through testLoader
            with torch.no_grad():

                # extracting the images from the batch
                val_images = val_batch['image'].to('cuda')
                val_images = torch.tensor(val_images, dtype=torch.float32).to('cuda')

                # passing the images through the model and the results through the resnet
                val_image_recon, val_z, mu, log_var, val_kld_loss = model(val_images)

                # calculating the features of the correspondig Conv2d-layers of the vgg-net
                x_features = extract_features(vgg, val_images, model='')
                x_recon_features = extract_features(vgg, val_image_recon, model='')

                # calculating the losses; recon_loss only for convenience reasons
                val_flp_loss = flp_loss_function.compute_loss(x_features, x_recon_features, val_images, val_image_recon)
                val_mse = F.mse_loss(val_image_recon, val_images, reduction='mean')

                          
                val_kl_sum = 0
                for kl_loss in val_kld_loss:
                    val_kl_sum += kl_loss
                    if (j% 200 == 0):
                        print("VAL_KL_LOSS: ",val_kl_sum)
                if dataset == 'Shape':
                    val_loss = val_flp_loss+shape_linear_annealing_loss(i, epoch, val_kl_sum, len(val_data), batch_size, factor)+val_mse
                else:
                    val_loss = val_flp_loss+style_linear_annealing_loss(i, epoch, val_kl_sum, len(val_data), batch_size, factor)+val_mse
              
                if j == 100:
                    show_progress(val_images, val_image_recon, val_z, model, epoch+1000, path, save = True)

            #account for possible nan-valkues if images cannot be reconstructed (rare case)
            if not val_loss.isnan().any():
                val_total_loss += val_loss.item()
                val_total_kld += val_kl_sum.item()
                count += 1
                val_total_flp += val_flp_loss.item()
                val_total_recon += val_mse.item()
            else:
                nan_count+=1
        torch.cuda.empty_cache()

        val_storer.set_values(epoch, val_total_loss / count, val_total_recon / count,
                              val_total_flp / count, val_total_kld / count, nan_count)
            
        train_storer.set_values(epoch, total_loss / len(trainLoader), total_recon / len(trainLoader),
                              total_flp / len(trainLoader), total_kld / len(trainLoader), 0)
        epoch += 1
