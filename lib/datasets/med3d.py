import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

csv_dir = '/home/xyh/data/PreData/LUNA16/extra_input/lung_seg'

train_black_list = [
'subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.243094273518213382155770295147',
'subset2/1.3.6.1.4.1.14519.5.2.1.6279.6001.172845185165807139298420209778',
'subset2/1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286',
'subset3/1.3.6.1.4.1.14519.5.2.1.6279.6001.603166427542096384265514998412',
'subset5/1.3.6.1.4.1.14519.5.2.1.6279.6001.397202838387416555106806022938',
'subset6/1.3.6.1.4.1.14519.5.2.1.6279.6001.251215764736737018371915284679',
'subset6/1.3.6.1.4.1.14519.5.2.1.6279.6001.167237290696350215427953159586',
'subset6/1.3.6.1.4.1.14519.5.2.1.6279.6001.177888806135892723698313903329'
]
valid_black_list = ['subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.130438550890816550994739120843']

df_train = pd.read_csv(os.path.join(csv_dir, 'train.csv'))
train_list = df_train['Id'].values.tolist()
for i in range(len(train_black_list)):
    train_list.remove(train_black_list[i])

df_valid = pd.read_csv(os.path.join(csv_dir, 'valid.csv'))
valid_list = df_valid['Id'].values.tolist()
valid_list.remove(valid_black_list[0])

class MED3D_dataset(Dataset):
    def __init__(self, split, data_dir):
        self.split = split
        self.data_dir = data_dir
        if self.split == 'trn':
            self.list = train_list
        elif self.split == 'val':
            self.list = valid_list

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        image_name = self.list[index].split('/')[1]
        image = np.load(os.path.join(self.data_dir, image_name+'_data.npy'))

        if image.shape[1]%32 != 0 or image.shape[2]%32 != 0:
            pad1 = 32-image.shape[1]%32
            pad2 = 32-image.shape[2]%32
            image = np.pad(image, ((0,0),(int(pad1/2),pad1-int(pad1/2)),(int(pad2/2),pad2-int(pad2/2))), mode='constant', constant_values=0)
        if self.split != 'test':
            mask = np.load(os.path.join(self.data_dir,image_name+'_mask.npy')) # (z, y, x)
            if mask.shape[1]%32 != 0 or mask.shape[2]%32 != 0:
                pad1 = 32-mask.shape[1]%32
                pad2 = 32-mask.shape[2]%32
                mask = np.pad(mask, ((0,0),(int(pad1/2),pad1-int(pad1/2)),(int(pad2/2),pad2-int(pad2/2))), mode='constant', constant_values=0)

        self.input_shape = image.shape

        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()