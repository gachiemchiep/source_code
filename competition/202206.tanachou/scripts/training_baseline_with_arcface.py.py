#!/usr/bin/env python
# coding: utf-8

# In[ ]:


imsize = 224
batch_size = 64
n_epochs = 30
num_workers = 8
init_lr = 1e-4
eval_id = 90
modelname = "efficientnet_b0"
use_amp = True
    
FEATS = 1536
FOLDS = 5


# In[ ]:


import time
import os
import numpy as np
import pandas as pd
import cv2
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler

import albumentations
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torchvision
import timm
from scipy import spatial
from sklearn.preprocessing import normalize
from sklearn import preprocessing


# In[ ]:


# Fix SEED
torch.manual_seed(47)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(47)


# In[ ]:


# Load labels
json_open = open('../data/train_meta.json', 'r')
df_train = pd.DataFrame(json.load(json_open)).T
df_train.reset_index(inplace=True, drop=True)

import glob
dfs = []
df_train["ids"] = -1
df_train["imdir"] = -1
for i in df_train.index:
    num = len(glob.glob(os.path.join("../data/train", str(i).zfill(3), "*")))
    for ii,_ in enumerate(range(num)):
        row = df_train.iloc[i]
        row.ids = ii
        row.imdir = i
        dfs.append(row)

df_train = pd.DataFrame(dfs)

# split with image_id
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=47)
df_train.reset_index(inplace=True)
df_train['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y=df_train.imdir.tolist())):
    df_train.loc[val_idx, 'fold'] = fold
df_train

# For CV
df_train2 = df_train[df_train.imdir<eval_id]
df_train_eval = df_train[df_train.imdir>=eval_id]
df_train


# In[ ]:


from torch.nn.parameter import Parameter
import math

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
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

        return output, output


# In[ ]:


class net(nn.Module):
    def __init__(self, modelname, out_dim=122, freeze_bn=True):
        super(net, self).__init__()
        self.modelname = modelname
        
        # for timm
        self.model = timm.create_model(modelname, pretrained=True)
        self.model.reset_classifier(0)
        model = timm.create_model(modelname, pretrained=False)
        model.reset_classifier(0)
        feats = int(model(torch.randn(1,3,imsize,imsize)).size()[-1])
        
        # Define loss
        self.arcface = ArcMarginProduct(512, out_dim) 

        # Embeddings
        self.fc = nn.Linear(feats, 512)
    
    # For evaluation
    def extract(self, x):
        return self.fc(self.model(x))
    
    # For training
    def forward(self, x, label):
        x = self.model(x)
        feat = self.fc(x).float()
        if self.training:
            x = self.arcface(feat, self.training, label)
        else:
            x = self.arcface(feat, self.training)
        return x


# In[ ]:


class buzaiDataset(Dataset):
    def __init__(self,
                 df,
                 test=False,
                 transform=None,
                ):

        self.df = df.reset_index(drop=True)
        self.test = test
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        file = os.path.join("../data/train", str(row.imdir).zfill(3), "{}.jpg".format(row.ids))
        images = cv2.imread(file)
        
        # Load labels
        label = row.imdir
        
        # aug
        if self.transform is not None:
            images = self.transform(image=images)['image']
        
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)
        
        label = label.astype(np.float32)
        return torch.tensor(images), torch.tensor(label)


# In[ ]:


# For embeddings
class buzaiDataset2(Dataset):
    def __init__(self,
                 df,
                 test=False,
                 transform=None,
                ):

        self.df = df.reset_index(drop=True)
        self.test = test
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        tiff_file = os.path.join("../data/train", str(row.imdir).zfill(3), "{}.jpg".format(row.ids))
        images = cv2.imread(tiff_file)
        
        # Load labels
        label = row.imdir
        
        # aug
        if self.transform is not None:
            images = self.transform(image=images)['image']
        
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)
        
        label = label.astype(np.float32)
        return torch.tensor(images), torch.tensor(row.imdir), torch.tensor(row.ids)


# In[ ]:


# いつもの
import albumentations as A
transforms_train = albumentations.Compose([
    albumentations.ShiftScaleRotate(scale_limit=0.3, rotate_limit=180,p=0.5),
    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                             val_shift_limit=0.2, p=0.5),
       A.RandomBrightnessContrast(brightness_limit=0.2, 
                                  contrast_limit=0.2, p=0.5),
    ],p=0.9),
    A.Cutout(num_holes=12, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
    albumentations.Rotate(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),   
    albumentations.Resize(imsize, imsize, p=1.0), 
])
transforms_val = albumentations.Compose([albumentations.Resize(imsize, imsize, p=1.0)])


# In[ ]:


dataset = buzaiDataset(df_train, transform=transforms_train)
# Setup dataloader
loader = torch.utils.data.DataLoader(dataset, batch_size=3, sampler=RandomSampler(dataset), num_workers=num_workers)
for (d,_) in loader:
    plt.imshow(d[0].transpose(0, 2).numpy())
    break


# In[ ]:


scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # for tensorcore engine

def train_epoch(loader, optimizer):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        data, target = data.to(device), target.to(device).long()
        loss_func = criterion
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _ = model(data, target)
            logits = logits.squeeze(1)
            loss = loss_func(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
    return np.mean(train_loss)


# In[ ]:


def val_epoch(loader, get_output=False):
    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target.to(device).long()
            _, logits = model(data, target)
            logits = logits.squeeze(1)
            loss_func = criterion
            loss = loss_func(logits, target)

            pred = logits.sigmoid().detach()
            LOGITS.append(logits)
            PREDS.append(pred)
            TARGETS.append(target)

            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy().argmax(1)
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    acc = np.sum(PREDS == TARGETS)/len(PREDS)*100
    
    if get_output:
        return LOGITS
    else:
        return val_loss, acc


# In[ ]:


def predict(loader, idx, model):
    embeddings = np.zeros((len(idx), 512))
    ids = []
    i = 0
    models = [model]

    # Inference
    with torch.no_grad():
        for (data, imdirs, idb) in tqdm(loader):
            data = data.to(device)
            
            # Infer for k-fold models
            embs = np.zeros((len(models), len(data), 512))
            for ii, model in enumerate(models):
                embs[ii] = model.extract(data).cpu().numpy()
            embs = embs.mean(axis=0)
            
            # collect embeddings and ids
            for emb in embs:
                embeddings[i] += emb
                i += 1
            for imdir, idb in zip(imdirs, idb):
                ids.extend([str(imdir.numpy().astype(int))])

    # Norm
    embeddings = normalize(embeddings, axis=1)
    return embeddings, ids


# In[ ]:


def getmap(IDX, train_embeddings, val_embeddings, train_ids):
    # mAP指標を計算
    distances = spatial.distance.cdist(val_embeddings[np.newaxis, IDX, :], train_embeddings, 'cosine')[0] # 0に近いほど似てる
    df_dist = pd.DataFrame(sorted([(train_ids[p], distances[p]) for p in range(len(distances))], key=lambda x: x[1]),columns=["ids", "dist"])

    results = []
    resultids = []
    TOPK = 10
    for id in sorted(df_dist.ids.unique()):
        results.append(df_dist[(df_dist.ids==id)].dist.mean())
        resultids.append(id)
    partition = np.argpartition(np.array(results), TOPK)[:TOPK]
    nearest = sorted([(resultids[p], results[p]) for p in partition], key=lambda x: x[1])

    SCORE = np.arange(1, 0., -0.1)
    try:
        score = SCORE[np.where(np.array(nearest)[:,0]==val_ids[IDX])[0][0]]
    except:
        score = 0
    return score


# In[ ]:


from timm.scheduler import CosineLRScheduler
folding = range(FOLDS)
device = "cuda"
criterion = nn.CrossEntropyLoss()

kernel_type = "{}-{}-arcface-val".format(modelname, imsize)

for fold in folding:
    # Setup dataset
    train_idx = np.where((df_train2['fold'] != fold))[0]
    valid_idx = np.where((df_train2['fold'] == fold))[0]
    df_this  = df_train2.loc[train_idx]
    df_valid = df_train2.loc[valid_idx]
    dataset_train = buzaiDataset(df_this, transform=transforms_train)
    dataset_valid = buzaiDataset(df_valid, transform=transforms_val, test=True)

    # Setup dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=RandomSampler(dataset_train), num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, sampler=SequentialSampler(dataset_valid), num_workers=num_workers)

    # setup embeddings
    valid_idx = np.where((df_train['fold'] == fold)*(df_train.imdir>=eval_id))[0]
    train_idx2 = np.where((df_train['fold'] != fold)*(df_train.imdir>=eval_id))[0]
    df_valid = df_train.loc[valid_idx]
    dataset_train = buzaiDataset2(pd.concat([df_this,df_train.loc[train_idx2]]), transform=transforms_val, test=True)
    train_loader2 = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=SequentialSampler(dataset_train), num_workers=num_workers)
    dataset_valid = buzaiDataset2(df_valid, transform=transforms_val, test=True)
    valid_loader2 = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, sampler=SequentialSampler(dataset_valid), num_workers=num_workers)  

    # Initialize model
    model = net(modelname, eval_id)
    model = model.to(device)

    print(len(dataset_train), len(dataset_valid))

    # We use Cosine annealing LR scheduling
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = CosineLRScheduler(optimizer, t_initial=n_epochs, lr_min=1e-6, 
                          warmup_t=3, warmup_lr_init=1e-5, warmup_prefix=True)

    best_file = f'./models/{kernel_type}_best_fold{fold}.pth'
    best = 0
    
    for epoch in range(1, n_epochs+1):
        start = time.time()
        torch.cuda.empty_cache() 
        print(time.ctime(), 'Epoch:', epoch)
        scheduler.step(epoch-1)

        # Train/val
        train_loss = train_epoch(train_loader, optimizer)
        val_loss, acc  = val_epoch(valid_loader)

        # predict embeddings
        train_embeddings, train_ids = predict(train_loader2, np.concatenate([train_idx2,train_idx]), model)
        val_embeddings, val_ids = predict(valid_loader2, valid_idx, model)
        scores = []
        for IDX in range(len(val_ids)):
            scores.append(getmap(IDX, train_embeddings, val_embeddings, train_ids))
        scores = np.mean(scores)
        print("map:", scores)

        # Log
        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, acc: {(acc):.5f}'
        print(content)
        with open(f'log/log_{kernel_type}_fold{fold}.txt', 'a') as appender:
            appender.write(content + '\n')

        # save only best model
        if scores > best:
            torch.save(model.state_dict(), os.path.join(f'models/{kernel_type}_fold{fold}.pth'))
            best = scores
#         break


# In[ ]:




