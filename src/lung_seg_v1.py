import os
import torch
import numpy as np

from torch.utils.data import Dataset

def get_customized_dataloader(split, ids_list, transform=None, **kwargs):

    is_train = split == 'trn'
    batch_size = kwargs['batch_size']
    num_workers = kwargs['num_workers']
    data_dir = kwargs['data_dir']

    dataset = AdjacentSliceDataset_v1(split, data_dir, ids_list, transform)

    return DataLoader(dataset,
                      shuffle=is_train,
                      batch_size=batch_size,
                      drop_last=is_train,
                      num_workers=num_workers,
                      pin_memory=True)

class AdjacentSliceDataset_v1(Dataset):
    """
    Dataset provides three adjacent slices of a case CT for the 2D DenseUnet
    """
    def __init__(self,
                 split,
                 data_dir,
                 ids_list,
                 transform=None,
                 **_):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.ids_list = ids_list
        self.id_index_list = np.array(range(len(self.ids_list)))
        self.current_id_index = 0
        self.transform = transform
        self.load_next_image()

    def load_next_image(self):
        self.image_name = os.path.basename(self.ids_list[self.id_index_list[self.current_id_index]])
        self.image = np.load(os.path.join(self.data_dir,self.image_name+'_data.npy')) # (z, y, x)
        if self.image.shape[1]%32 != 0 or self.image.shape[2]%32 != 0:
            pad1 = 32-self.image.shape[1]%32
            pad2 = 32-self.image.shape[2]%32
            self.image = np.pad(self.image, ((0,0),(int(pad1/2),pad1-int(pad1/2)),(int(pad2/2),pad2-int(pad2/2))), mode='constant', constant_values=0)
        if self.split != 'test':
            self.masks = np.load(os.path.join(self.data_dir,self.image_name+'_mask.npy')) # (z, y, x)
            if self.masks.shape[1]%32 != 0 or self.masks.shape[2]%32 != 0:
                pad1 = 32-self.masks.shape[1]%32
                pad2 = 32-self.masks.shape[2]%32
                self.masks = np.pad(self.masks, ((0,0),(int(pad1/2),pad1-int(pad1/2)),(int(pad2/2),pad2-int(pad2/2))), mode='constant', constant_values=0)
        self.nums = int(len(self.image))
        self.input_shape = (3, self.image.shape[1], self.image.shape[2])

        self.current_id_index += 1
        if self.current_id_index >= len(self.ids_list):
            self.current_id_index = 0
            self.id_index_list = np.random.permutation(len(self.ids_list))

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        input_img = 170*np.ones(self.input_shape, dtype=np.float32) # (3, 224, 224)
        if self.split != 'test':
            input_mask1 = np.zeros(self.input_shape, dtype=np.float32)  # (3, 224, 224)
            input_mask2 = np.zeros(self.input_shape, dtype=np.float32)  # (3, 224, 224)
        if index == 0:
            input_img[1:,...] = self.image[0:2, ...]
            if self.split != 'test':
                input_mask1[1:,...] = (self.masks[0:2, ...]==1).astype('float32')
                input_mask2[1:,...] = (self.masks[0:2, ...]==2).astype('float32')
        elif index == int(len(self.image)-1):
            input_img[0:2,...] = self.image[index-1:index+1, ...]
            if self.split != 'test':
                input_mask1[0:2,...] = (self.masks[index-1:index+1, ...]==1).astype('float32')
                input_mask2[0:2,...] = (self.masks[index-1:index+1, ...]==2).astype('float32')
        else:
            input_img = self.image[index-1:index+2, ...]
            if self.split != 'test':
                input_mask1 = (self.masks[index-1:index+2, ...]==1).astype('float32')
                input_mask2 = (self.masks[index-1:index+2, ...]==2).astype('float32')
        input_img = input_img/255.

        if self.split != 'test':
            return torch.from_numpy(input_img).float(), torch.from_numpy(input_mask1).float(), torch.from_numpy(input_mask2).float()
        else:
            return torch.from_numpy(input_img).float()
        

