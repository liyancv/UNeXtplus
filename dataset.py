# 这是一个用于PyTorch的自定义数据集类 Dataset，用于加载图像分割任务的数据。
import os
import cv2
import numpy as np
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    # 初始化函数，接收图像和mask的文件路径，文件扩展名，类别数以及可选的图像变换。
    # img_ids 是图像的标识符列表，
    # img_dir 是包含图像文件的目录，
    # mask_dir 是包含掩膜文件的目录，
    # img_ext 和 mask_ext 是图像和掩膜文件的扩展名，
    # num_classes 是类别数，
    # transform 是一个可选的图像变换，用于数据增强等。
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """    
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

#     返回数据集的长度，即图像数量。
    def __len__(self):
        return len(self.img_ids)
#     加载单个图像及其对应的mask
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        
        img = cv2.imread(img_path)

        mask = []
        for i in range(self.num_classes):
            path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)           
            mask.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            # mask = augmented['label']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}
