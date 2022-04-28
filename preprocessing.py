#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import urllib
import urllib.request
import json
import uuid
import os
import cv2

def preprocessMVC():
    with open('./data/image_links.json') as data:
        data = json.load(data) 
    with open('./data/attribute_labels.json') as labels:
        labels = json.load(labels)
    with open('./data/mvc_info.json') as info:
        info = json.load(info)

    df = pd.json_normalize(info)
    df2 = pd.json_normalize(labels)
    df3 = pd.concat([df,df2], 1)
    df3['URL'] = data
    
    #remove the views that we dont want (especially from the back)
    df4 = df3[(df3.viewId == 3) | (df3.viewId == 0) | (df3.viewId == 1)]
    
    #Only one category to reduce variance in the training images
    df5 = df4[df4.category == '"Shirts & Tops"']
    
    #Drop the values not needed
    df6 = df5.drop(['colourId','productTypeId','category','fiftyU','hundred1U','hundred2U','hundred2O','zetaCategory','image_4x_width','image_4x_height','image_url_2x',
                'total_style','brandId','image_url_thumbnails','price','productName','catNum','styleId','image_url_multiView'],1)
    
    df6['itemN'] = df6.apply(lambda row: int(row['itemN']), axis=1)
    
    #Retrieve the images in a for_loop
    for index, row in df6.iterrows():  
        try:
            urllib.request.urlretrieve(row['image_url_4x'], './preprocessed_data_shapes/'+str(int(row['itemN']))+'_'+str(row['viewId'])+'.jpg')
        except urllib.request.HTTPError:
            pass

        
def preprocessImageNet():
    #set directory path to the path of the donwloaded imagenet data
    #set out_dir to the preferred directory for the cropped data; "preprocessed_data_styles" is the default path for the dataloader
    directory = './image_val/'
    out_dir = './preprocessed_data_styles/'

    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".JPEG"):
            img = cv2.imread(directory + filename)
            resized = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
            cv2.imwrite(out_dir+filename, resized)
            count += 1
            continue
        else:
            continue
    print(count)

