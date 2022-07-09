from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import torch.nn as nn


'''NOTE:
This script is provided to make the evaluation process more transparent. It does not work without prior dataset 
generation. We provide an exemplified dataloader for training the model for the sleeve lengths classifier.'''

'''INSTRUCTIONS for executing this script:
In order to train the model, a dataloader has to be defined that uses the filtered original garment data. 
Once the dataset and dataloader are created, the model can be trained using the train() method.

To test the model, create a dataset with attribute conditioned garment images, establish a dataloader, and call the test() method.'''


class MyDataset(Dataset):
    """Dataloader class to load the images batch per batch
        --------------
        Inspired by the official tutorial https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        Changed to fit the corresponding datasets
    """
    def __init__(self, root_dir, att_type, transform=None, attribute_df=None):
        self.attributes_frame = attribute_df
        self.root_dir = root_dir
        self.transform = transform
        self.att_type = att_type

    def __len__(self):
        return len(self.attributes_frame)

    def __getitem__(self, itemId):
        """Method that iterates over the csv-data in order to load images batch per batch.
            Customized to fit the different datasets. Method loads the data based on the
            filename attribute in the csv file. Attributes are not used as no conditional VAE is implemented.
        """

        if torch.is_tensor(itemId):
            itemId = itemId.tolist()

        if self.att_type == 'len':
            img_name = os.path.join(self.root_dir, str(self.attributes_frame['itemN'].iloc[itemId]) +"_"+str(self.attributes_frame['viewId'].iloc[itemId])+".jpg")
            attributes = self.attributes_frame.iloc[itemId, 2:]
   
        image = io.imread(img_name)

        attributes = np.array(attributes, dtype=np.int64)
        sample = {'image': image, 'attributes': attributes}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Class that rescales the images based on the transformation.
    Each batch is resized when loaded by simple transform.resize.
    Simplified implementation as images are always resized to static values.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, attributes = sample['image'], sample['attributes']

        new_h, new_w = self.output_size, self.output_size

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'attributes': attributes}

class ToTensor(object):
    """Method that converts the numpy image to tensors and swaps the axis
    """

    def __call__(self, sample):
        image, attributes = sample['image'], sample['attributes']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'attributes': torch.from_numpy(attributes)}

def sample_dataset(img_size=128):
    """Sample dataloader for 3 different types of sleeve length. For our evaluation model, we furthermore filtered for unique attributes to reduce the class overlap. Note that we also split the data previously to gain a test and a training set."""
    
    df_att = pd.read_csv('./data/preprocessed_shapes_metadata.csv')
    df_att['NoSleeves'] = df_att[['Sleeveless','TankTops']].max(1)
    return MyDataset(root_dir='./data/preprocessed_data_shapes/',
                     att_type = 'len',
                     attribute_df=df_att[['itemN','viewId','LongSleeves','ShortSleeves','NoSleeves']],
                     transform=transforms.Compose([
                         Rescale(img_size),
                         ToTensor()
                     ]))




class Swish(nn.Module):
    """Swish activation function implemented by Vahdat & Kanutz (2020), "NVAE".
        Multiplies x with x after a sigmoid activation.
        Parameters
        ----------
        x: image/filter tensor that is passed through the network

        Implementation References
        --------------
        Taken from https://github.com/NVlabs/NVAE
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

#This class describes the model with a resnet backbone and a linear classification head.
class classNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.fnn = nn.Sequential(
            nn.Linear(1000, 512), 
            nn.BatchNorm1d(512), Swish(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128), Swish(),
            nn.Linear(128, output_dim),
            nn.Softmax()
        )
  
    def forward(self, x):
        return self.fnn(self.backbone(x))


#Train the standard malignancy prediction model
def train(batch_size=24, epochs=50):
    """Method to train the model, splitting the remaining data (after the manual train/test split) into a train and a validation set.
    Set the paths at the corresponding positions to run the code in your environment.
    """
    
    #The following lines of code need to be replaced once a dataloader is created
    dataset = sample_dataset()
    lengths = [int(len(dataset)*0.9)+1, int(len(dataset)*0.1)]
    train_data, val_data = random_split(dataset, lengths)
    
    trainLoader = DataLoader(train_data, batch_size, shuffle=True)
    valLoader = DataLoader(val_data, batch_size, shuffle=True)
    
    #loss definition
    criterion = nn.CrossEntropyLoss()
    
    #define model and optimizer, set the number of output neurons as needed
    model = classNet(3).to('cuda')
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
    cp_epoch = 0

    #load model from checkpoint, change the path as needed
    #cp = torch.load('PATH')
    #model.load_state_dict(cp['model_state_dict'])
    #model_optimizer.load_state_dict(cp['optimizer_state_dict'])
    #cp_epoch = cp['epoch']
    criterion = nn.CrossEntropyLoss()
    avg_loss = 0
    
    for epoch in range(cp_epoch,epochs):
        #iterate batchwise through trainloader
        print("Epoch: ", epoch)
        model.train()
        avg_loss = 0
        for i, batch in enumerate(trainLoader):
            #print("batch: ",batch)

            # extracting the images from the batch
            images = batch['image']
            images = torch.tensor(images, dtype=torch.float32)
            images = images.to('cuda')

            labels = batch['attributes'].to('cuda')
            out = model(images)

            loss = criterion(out,labels.float())

            avg_loss += loss
            if i % 10 == 5:
                print("AVG_loss: ",avg_loss.item() /(i+1))

            loss.backward()
            model_optimizer.step()
        
        
        avg_val_loss = 0  
        TP = 0
        model.eval()    
        with torch.no_grad():
            for j, batch in enumerate(valLoader):
                    
                # extracting the images from the batch
                images = batch['image']
                images = torch.tensor(images, dtype=torch.float32)
                images = images.to('cuda')

                labels = batch['attributes'].to('cuda')

                out = model(images)

                loss = criterion(out,labels.float())

                pred_classes = torch.argmax(out, dim=1)
                actual_classes = torch.argmax(labels, dim=1)

                TP += (pred_classes == actual_classes).float().sum()

                avg_val_loss += loss
                if j % 20 == 10:
                    print("AVG_val_loss: ",avg_val_loss.item() /((j+1)*batch_size))  
                
            print(TP)
        
            print("FINAL: ", TP/len(val_data))    

            print("\nTRAINING LOSS: ",avg_loss.item()/(len(trainLoader)*batch_size),"\t\tVAL LOSS: ",avg_val_loss.item()/(len(valLoader)*batch_size))    

        #Set a path here
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model_optimizer.state_dict(),
            'loss': loss,
        }, "PATH.tar".format(epoch))  

def test(batch_size=64):
    """Method to test the model on the a) test set of the original data and b) the attribute conditioned, generated data. 
    Returns the accuracy of the model. 
    """

    #define a test dataset as needed
    dataset = sample_dataset()
    testLoader = DataLoader(dataset, batch_size, shuffle=False)
    
    #loss definition
    criterion = nn.CrossEntropyLoss()
    
    #define model
    model = classNet(3)

    #load model from checkpoint
    #Set path to your saved weights here
    cp = torch.load('PATH', map_location=torch.device('cpu'))
    model.load_state_dict(cp['model_state_dict'])
    
    criterion = nn.CrossEntropyLoss()
    avg_loss = 0
    
    #iterate batchwise through trainloader
    model.eval()
    with torch.no_grad():
        TP = 0
        for i, batch in enumerate(testLoader):
            images = batch['image']
            images = images

            labels = batch['attributes']
            out = model(images)

            pred_classes = torch.argmax(out, dim=1)
            actual_classes = torch.argmax(labels, dim=1)

            TP += (pred_classes == actual_classes).float().sum()

            print(pred_classes)
            print(actual_classes)
            print(TP)
    
    print("FINAL: ", TP/len(dataset))
    
train()