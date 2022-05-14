import os
import numpy as np
import torch
import torch_geometric.data as gdata

from torch_geometric.data import Data
from DataUtils.MarginExtraction import getMargin
from augmentations.transforms import Compose, GaussianNoise, Flip, \
    PadIfNeeded, RandomRotate90, RandomScale2,  \
     Resize, ElasticTransform, RandomGamma, Rotate
import SimpleITK as sitk

class DatasetBuild(gdata.Dataset):
    def __init__(self, root, label,usefile, Is_train=True,transform=None, pre_transform=None):
        super(DatasetBuild, self).__init__(root, transform, pre_transform)
        self.usefile = usefile  ## image name
        self.label = label  ## label arrary
        self.root = root   ## image path
        self.Is_train = Is_train  ## the flage that whether is training, which will be applied data augmentation

        if self.Is_train:
            ## data augumentation
            self.aug = Compose([
                Flip(axis=0, p=0.5),
                Flip(axis=1, p=0.5),
                Flip(axis=2, p=0.5),
                GaussianNoise(var_limit=(0, 0.3), mean=0, p=0.5),
                RandomScale2(scale_limit=[0.7, 1.0], p=0.5),
                Rotate(p=0.5),
                ElasticTransform(p=0.5),
                RandomRotate90(p=0.5),
                RandomGamma(gamma_limit=(0.7, 1.5), p=0.5)
            ])

    @property
    def raw_dir(self):
        pass

    @property
    def processed_dir(self):
        pass


    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def len(self):
        return len(self.usefile)

    def process(self):
        pass

    def _process(self):
        pass

    def extract_largest_image_mask(self, image, mask):
        x = np.sum(np.sum(mask, axis=1), axis=1)
        y = np.sum(np.sum(mask, axis=0), axis=1)
        z = np.sum(np.sum(mask, axis=0), axis=0)
        indexx, indexy, indexz = np.argmax(x), np.argmax(y), np.argmax(z)
        UseSlice = []
        UseSlice.append([image[indexx], mask[indexx]])
        UseSlice.append([image[:, indexy], mask[:, indexy]])
        UseSlice.append([image[:, :, indexz], mask[:, :, indexz]])
        return UseSlice

    def build_graph(self, curve, y):
        curve = torch.from_numpy(curve).float()
        head = np.expand_dims(np.asarray(range(curve.size(0))), 0)
        end = head + 1
        end[:, -1] = head[:, 0]
        left = np.concatenate([head, end], axis=-1)
        right = np.concatenate([end, head], axis=-1)
        adj = np.concatenate([left, right], axis=0)
        adj = torch.from_numpy(adj).long()
        data = Data(x=curve, edge_index=adj, y=y)
        return data


    def get(self, idx):
        newname = self.usefile[idx]

        img = sitk.ReadImage(os.path.join(self.root, 'image'), newname)  ## image file path
        mask = sitk.ReadImage(os.path.join(self.root, 'mask'), newname)  ## mask file path

        img = sitk.GetArrayFromImage(img)
        mask = sitk.GetArrayFromImage(mask)


        if self.Is_train:
            ## data augumentation
            data_ = {'image': img, 'mask': mask}
            data_ = self.aug(**data_)

            ## insure that tha mask would not change to zeros after data augmentation
            if np.sum(data_['mask']) == 0:
                img = img
                mask = mask
            else:
                img = data_['image']
                mask = data_['mask']

        ## extract 2D slice of three views from 3D volume
        useslice = self.extract_largest_image_mask(img, mask)

        ## extract margin and build graph
        gcndata = {}
        for i, SubSlice in enumerate(useslice):
            curve = getMargin(SubSlice[0], SubSlice[1], pointN=15, stride=1)
            curve[np.isnan(curve)] = 0
            gcndata['data{}'.format(i+1)] = self.build_graph(curve=curve, y=self.label[idx])

        ## keep 3D image
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return gcndata, img, mask



if __name__ =='__main__':

    trainset = DatasetBuild(root='./dataset', label= np.asarray([0,1]),usefile=['subject0.nii.gz', 'subject1.nii.gz'], Is_train=True)
    trainloader = gdata.DataLoader(trainset, batch_size=2, shuffle=True, pin_memory=True, num_workers=2)

    for i, [gcndata, img, mask] in enumerate(trainloader):
        data1 = gcndata['data1']
        data2 = gcndata['data2']
        data3 = gcndata['data3']
        y = gcndata['data1'].y
