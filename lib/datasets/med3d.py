import os
import torch
import numpy as np
import pandas as pd
from scipy import ndimage

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
    def __init__(self, split, data_dir, 
                 input_D=256, 
                 input_H=256, 
                 input_W=256):
        self.split = split
        self.data_dir = data_dir
        if self.split == 'trn':
            self.list = train_list
        elif self.split == 'val':
            self.list = valid_list
        self.input_D = input_D
        self.input_H = input_H
        self.input_W = input_W

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        image_name = self.list[index].split('/')[1]
        image = np.load(os.path.join(self.data_dir, image_name+'_data.npy'))

        if image.shape[1]%32 != 0 or image.shape[2]%32 != 0:
            pad1 = 32-image.shape[1]%32
            pad2 = 32-image.shape[2]%32
            image = np.pad(image, ((0,0),(int(pad1/2),pad1-int(pad1/2)),(int(pad2/2),pad2-int(pad2/2))), mode='constant', constant_values=0)
        if self.split != 'tst':
            mask = np.load(os.path.join(self.data_dir,image_name+'_mask.npy')) # (z, y, x)
            if mask.shape[1]%32 != 0 or mask.shape[2]%32 != 0:
                pad1 = 32-mask.shape[1]%32
                pad2 = 32-mask.shape[2]%32
                mask = np.pad(mask, ((0,0),(int(pad1/2),pad1-int(pad1/2)),(int(pad2/2),pad2-int(pad2/2))), mode='constant', constant_values=0)

        self.input_shape = image.shape
#         image = self.__resize_data__(image)/255
#         mask = self.__resize_data__(mask)/255
        
        image, mask = self.__training_data_process__(image, mask)        
        image, mask = image[np.newaxis,...], mask[np.newaxis,...]
        
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()
    # TODO: complete split == 'tst'
#         elif self.phase == "test":
#             # read image
#             ith_info = self.img_list[idx].split(" ")
#             img_name = os.path.join(self.root_dir, ith_info[0])
#             print(img_name)
#             assert os.path.isfile(img_name)
#             img = nibabel.load(img_name)
#             assert img is not None

#             # data processing
#             img_array = self.__testing_data_process__(img)

#             # 2 tensor array
#             img_array = self.__nii2tensorarray__(img_array)

#          return img_array

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)
        
        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        
        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]


    def __random_center_crop__(self, data, label):
        from random import random
        """
        Random crop
        """
        target_indexs = np.where(label>0)
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth*1.0/2) * random())
        Y_min = int((min_H - target_height*1.0/2) * random())
        X_min = int((min_W - target_width*1.0/2) * random())
        
        Z_max = int(img_d - ((img_d - (max_D + target_depth*1.0/2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height*1.0/2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width*1.0/2)) * random()))
       
        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])
 
        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)
        
        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]



    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data


    def __crop_data__(self, data, label):
        """
        Random crop with different methods:
        """ 
        # random center crop
        data, label = self.__random_center_crop__ (data, label)
        
        return data, label

    def __training_data_process__(self, data, label): 
        # crop data according net input size
        
        # drop out the invalid range
        data, label = self.__drop_invalid_range__(data, label)
        
        # crop data
        data, label = self.__crop_data__(data, label) 

        # resize data
        data = self.__resize_data__(data)
        label = self.__resize_data__(label)

        # normalization datas
        # data = self.__itensity_normalize_one_volume__(data)

        return data, label


    def __testing_data_process__(self, data): 
        # crop data according net input size
        data = data.get_data()

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data