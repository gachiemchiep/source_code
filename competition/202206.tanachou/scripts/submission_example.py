#!/usr/bin/env python
# coding: utf-8

# In[ ]:


IMSIZE = 224
MODELNAME = "efficientnet_b0"
batch_size = 16
num_workers = 0
FEATS = 512
DEBUG = True


# In[ ]:


import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
import albumentations
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torchvision
import timm
from scipy import spatial
from sklearn.preprocessing import normalize




from torch.nn.parameter import Parameter
import math
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, train, label=False):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight)).float()
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        #print(self.mm)
        #print(cosine)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        if train:
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot.scatter_(1, label.cuda().view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        else:
            output = cosine
        output *= self.s

        return output




class net(nn.Module):
    def __init__(self, modelname, out_dim=122, freeze_bn=True):
        super(net, self).__init__()
        
        self.model = timm.create_model(modelname, pretrained=True)
        self.model.reset_classifier(0)
        feats = FEATS
        self.arcface = ArcMarginProduct(feats, out_dim) 
        self.dropout = nn.Dropout(0.2)
        
    def predict(self, x):
        return self.model(x)

    def forward(self, x, label):
        feat = self.model(x).float()
        if self.training:
            x = self.arcface(feat, self.training, label)
        else:
            x = self.arcface(feat, self.training)
        return x




import albumentations as A
transforms_val = albumentations.Compose([albumentations.Resize(IMSIZE, IMSIZE, p=1.0)])




class buzaiDataset(Dataset):
    def __init__(self,
                 files,
                 transform=None,
                ):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        images = cv2.imread(file)

        # aug
        if self.transform is not None:
            images = self.transform(image=images)['image']
    
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)
        return torch.tensor(images)




def predict(loader, idx, model):
    embeddings = np.zeros((len(idx), FEATS))
    models = [model]
    i = 0

    # Inference
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to("cuda")
            
            # Infer for k-fold models
            embs = np.zeros((len(models), len(data), FEATS))
            for ii, model in enumerate(models):
                embs[ii] = model.predict(data).cpu().numpy()
            embs = embs.mean(axis=0) # average over kfolds
            # collect embeddings and ids
            for emb in embs:
                embeddings[i] += emb
                i += 1

    # Norm
    embeddings = normalize(embeddings, axis=1)
    return embeddings




def load_model(model_path):
    """
    load some model(s)
    """
    
    model = net(MODELNAME, 100)
    model = model.to("cuda")
    model.load_state_dict(torch.load(model_path))
    return model


def make_reference(reference_path, reference_meta, model):
    """
    make some features for reference data
    """
    # Make file dirs
    reference_dirs = list(reference_meta.keys())
    ids = []
    
    # Get file fullpath
    files = []
    import glob
    if DEBUG:
        for refdir in reference_dirs[:20]:
            foundfiles = glob.glob(os.path.join(reference_path, refdir, "*jpg"))
            files.extend(foundfiles)
            for _ in foundfiles:
                ids.append(refdir)
    else:
        for refdir in reference_dirs:
            foundfiles = glob.glob(os.path.join(reference_path, refdir, "*jpg"))
            files.extend(foundfiles)
            for _ in foundfiles:
                ids.append(refdir)
    dataset = buzaiDataset(files, transforms_val)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset), num_workers=num_workers)

    # Get embeddings
    embeddings = predict(loader, files, model)
    
    return embeddings, np.array(ids)


def read_image(input):
    """
    read some image
    """
    images = cv2.imread(input)
    images = transforms_val(image=images)['image']
    images = images.astype(np.float32)
    images /= 255
    images = images.transpose(2, 0, 1)
    return torch.tensor(images)

def postprocess(prediction, embeddings, ids):
    """
    post-process some prediction
    """
    distances = spatial.distance.cdist(prediction, embeddings, 'cosine')[0] # 0に近いほど似てる
    df_dist = pd.DataFrame(sorted([(ids[p], distances[p]) for p in range(len(distances))], key=lambda x: x[1]),columns=["ids", "dist"])

    results = []
    resultids = []
    TOPK = 10
    for id in sorted(df_dist.ids.unique()):
        results.append(df_dist[(df_dist.ids==id)].dist.mean())
        resultids.append(id)
    partition = np.argpartition(np.array(results), TOPK)[:TOPK]
    nearest = sorted([(resultids[p], results[p]) for p in partition], key=lambda x: x[1])
    return list(np.array(nearest)[:,0])




class ScoringService(object):
    @classmethod
    def get_model(cls, model_path, reference_path, reference_meta_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            reference_path (str): Path to the reference data.
            reference_meta_path (str): Path to the meta data.

        Returns:
            bool: The return value. True for success, False otherwise.
        """
        #try:
        cls.model = load_model(os.path.join(model_path, MODELNAME + "-{}-arcface-val_fold{}.pth".format(IMSIZE, 0)))
        with open(reference_meta_path) as f:
            reference_meta = json.load(f)
        embeddings, ids = make_reference(reference_path, reference_meta, cls.model)
        cls.embeddings = embeddings
        cls.ids = ids
        #    return True
        #except:
        #    return False

    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input (str): path to the image you want to make inference from

        Returns:
            dict: Inference for the given input.
        """
        # load an image and get the file name
        image = read_image(input)
        sample_name = os.path.basename(input).split('.')[0]

        # make prediction
        with torch.no_grad():
            prediction = cls.model.predict(image.unsqueeze(0).cuda()).cpu().numpy()

        # make output
        prediction = postprocess(prediction, cls.embeddings, cls.ids)
        output = {sample_name: prediction}

        return output




ScoringService.get_model('models', 'train', 'train_meta.json')
ScoringService.ids
ScoringService.predict('../data/train/000/0.jpg')

