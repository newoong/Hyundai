import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import pathlib
from tqdm import tqdm


class HyundaiLoader(object):
    def __init__(self, data_path, test_path, win_size, mode="train"):
        self.mode = mode
        self.win_size = win_size
        path=pathlib.Path(data_path)

        if mode=='train':

            print("Train_data loading...")
            train_list=list(path.glob('*train*.npy'))
            for idx,data in enumerate(tqdm(train_list)):
                if idx==0:
                    train_data=np.load(data)
                else:
                    temp=np.load(data)
                    train_data=np.concatenate((train_data,temp),axis=0)

            np.random.seed(10)
            np.random.shuffle(train_data)
            self.train = train_data
            print("train:", self.train.shape)

        elif mode=='val':
            print("Valid_data loading...")
            valid_list=list(path.glob('*valid*.npy'))
            for idx,data in enumerate(tqdm(valid_list)):
                if idx==0:
                    valid_data=np.load(data)
                else:
                    temp=np.load(data)
                    valid_data=np.concatenate((valid_data,temp),axis=0)
            self.valid = valid_data
            print("valid:", self.valid.shape)

        elif mode=='test':
            test_data = pd.read_parquet(test_path)
            # test_data.fillna(method='ffill', limit = 3, inplace=True)
            # test_data.dropna(axis=0, inplace=True)
            test_data.dropna(how='all',inplace=True)
            test_data.sort_index(inplace=True)
            test_data.fillna(method='ffill',inplace=True)
            test_data.fillna(method='bfill',inplace=True)
            # test_data.fillna(method='bfill',inplace=True)
            test_data, indexes = test_sliding_window_temp(test_data,win_size,win_size*5+int(0.1*(win_size*5)),60)
            self.test = test_data
            self.indexes = indexes
        #self.test_labels = np.load(data_path + "/test_label.npy")
        #self.test_labels=np.zeros(shape=(len(test_data.shape[0]*test_data.shape[1],)))


    def __len__(self):

        if self.mode == "train":
            return self.train.shape[0]
        elif (self.mode == 'val'):
            return self.valid.shape[0]
        elif (self.mode == 'test'):
            return len(self.test)

    def __getitem__(self, index):
        if self.mode == "train":
            return self.train[index]
        elif (self.mode == 'val'):
            return self.valid[index]
        elif (self.mode == 'test'):
            return self.test[index],np.zeros(shape=(len(self.test[index],)))



def test_sliding_window(df,window_size,threshold): #threshold(��): �������������� window������ ���?
        df.index=pd.to_datetime(df.index)
        windows = []
        start=0
        while start<len(df):
            window = df.iloc[start:start+window_size]
            time_idx=window.index
            gap=time_idx[-1]-time_idx[0]
            if gap.seconds>threshold:
                start+=1
                continue
            else:
                windows.append(window.values)
                start+=window_size
                
        #windows=np.array(windows,dtype=np.float32)
        return windows

def test_sliding_window_temp(df,window_size,threshold, stride): #threshold(��): �������������� window������ ���?
        df.index=pd.to_datetime(df.index)
        windows = []
        indexes = []
        for i in tqdm(range(0, len(df) - window_size + 1, stride)):
            j = i+window_size
            if j>len(df):
                break
            window = df.iloc[i:j]
            time_idx=window.index
            gap=time_idx[-1]-time_idx[0]
            if gap.seconds>threshold:
                continue
            else:
                windows.append(window.values)
                indexes.append(window.index)
                
        windows=np.array(windows,dtype=np.float32)
        return np.squeeze(windows), indexes



def get_loader_segment(data_path, test_path, dataset, batch_size, win_size=100, mode='train'):
    if dataset=='Hyundai':
        dataset = HyundaiLoader(data_path, test_path, win_size, mode)

    shuffle=False
    if mode=='train':
        shuffle=True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)

    return data_loader
