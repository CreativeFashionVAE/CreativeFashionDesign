###SEE AWESOME TUTORIAL: https://docs.lightly.ai/tutorials/package/tutorial_simclr_clothing.html
import sys
print(sys.executable)
import os
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image

import numpy as np
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
from torch.nn.modules.dropout import Dropout
from torch import nn

'''NOTE:
This script is provided to make the evaluation process more transparent. It does not work without prior dataset generation.'''

'''INSTRUCTIONS for executing this script:

First, create a simclr_data() object, setting parameters of choice. Path_to_data should link to the folder with the training images and test_path should link to the folder containing training AND test images. After creating an object of the class, the train-method and, subsequently, the generate_embeddings-method should be called.

Once the embeddings are created, the plot_knn_examples method can be executed. As parameters, a list of image names should be passed that contains the images of the test set that you want to compute the novelty for (chosen_list). Chosen_list_index should point to the corresponding index positions and prevents distance computation between images of the test set. n_neighbors defines the n closest neighbors to which the distance from each image of chosen_list is taken into account. 

The final result is the average distance from all images of chosen_list to their n nearest neighbors. More novel images will achieve a higher score.'''

class simclr_data():
    def __init__(self, save_path = './data/model1_novelty.pt', batch_size=128, seed=1, max_epochs=1, input_size=128, num_ftrs=32, path_to_data='./data/preprocessed_data_shapes', test_path='./data/preprocessed_data_shapes'):
        self.num_workers = 8
        self.batch_size = batch_size
        self.seed = seed
        self.max_epochs = max_epochs
        self.input_size = input_size
        self.num_ftrs = num_ftrs
        self.save_path = save_path

        pl.seed_everything(seed)
        self.path_to_data = path_to_data
        self.test_path= test_path

        collate_fn = lightly.data.SimCLRCollateFunction(
        input_size=input_size,
        vf_prob=0.5,
        rr_prob=0.5
        )

        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=lightly.data.collate.imagenet_normalize['mean'],
                std=lightly.data.collate.imagenet_normalize['std'],
            )
        ])

        self.dataset_train_simclr = lightly.data.LightlyDataset(
            input_dir=path_to_data
        )

        self.dataset_test = lightly.data.LightlyDataset(
            input_dir=test_path,
            transform=self.test_transforms
        )

        self.dataloader_train_simclr = torch.utils.data.DataLoader(
            self.dataset_train_simclr,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=self.num_workers
        )

        self.dataloader_test = torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers
        )

    def train(self):
        gpus = 1 if torch.cuda.is_available() else 0

        self.model = SimCLRModel(self.max_epochs)

        trainer = pl.Trainer(
            max_epochs=self.max_epochs, gpus=gpus, progress_bar_refresh_rate=100
        )

        trainer.fit(self.model, self.dataloader_train_simclr)
        torch.save(self.model, self.save_path)

    def generate_embeddings(self, load=False):

        if load:
            self.model = torch.load(self.save_path)

        self.model.eval()
        self.embeddings = []
        self.filenames = []
        with torch.no_grad():
            for img, label, fnames in self.dataloader_test:
                img = img.to(self.model.device)
                emb = self.model.backbone(img).flatten(start_dim=1)
                self.embeddings.append(emb)
                self.filenames.extend(fnames)

        self. embeddings = torch.cat(self.embeddings, 0)
        self.embeddings = normalize(self.embeddings)

    def get_image_as_np_array(self, filename: str):
        img = Image.open(filename)
        return np.asarray(img)

    def plot_knn_examples(self, chosen_list, chosen_list_index, n_neighbors=70):

        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)

        #samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)
        samples_idx=list()
        
        for file in chosen_list:
            samples_idx.append(self.filenames.index(file))
        
        avg_distance = 0
        for idx in samples_idx:
            fig = plt.figure(figsize=(20,30))
            
            distance = 0
            count=0
            
            for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
                ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
                fname = os.path.join(self.test_path, self.filenames[neighbor_idx])
                plt.imshow(self.get_image_as_np_array(fname))
                
                if neighbor_idx not in chosen_list_index:
                    distance += distances[idx][plot_x_offset]
                    count+=1
                    
                
                ax.set_title(f'd={distances[idx][plot_x_offset]:.3f}')
                plt.axis('off')
                
                if count == n_neighbors:
                    break
            
            avg_distance += distance/count
            print("avg_distance of {}: {}".format(idx, distance/count))
                    
        
        avg_distance = avg_distance/13
        print("final: ",avg_distance)
        return nbrs, distances, indices
        

'''Class for the garment shape classification model. Consits of resnet with classification head.'''
class SimCLRModel(pl.LightningModule):
    def __init__(self, max_epochs):
        super().__init__()

        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.max_epochs = max_epochs
        
        ##For Pretrained Weights of ResNet Model: 
        #ckpt = torch.load('model.pth')
        #self.backbone.load_state_dict(ckpt['resnet18_parameters'])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.max_epochs
        )
        return [optim], [scheduler]
