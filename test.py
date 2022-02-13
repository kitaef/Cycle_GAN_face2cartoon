accelerator = 'gpu'

import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.optim import Adam
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tt
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available()  else 'cpu'


class ImageDS(Dataset):

    def __init__(self, path_a, path_b, image_size):
        super().__init__()
        self.image_size = image_size
        self.images_a = self.process_folder(path_a, crop='face')
        self.images_b = self.process_folder(path_b, crop='cartoon')
        self.size_a = len(self.images_a)
        self.size_b = len(self.images_b)
        self.bigger = True if self.size_a >= self.size_b else False

    def process_folder(self, path, crop):
        images = []
        progress_bar = tqdm(total=DATASET_SIZE, desc='Images processed:', position=0)
        for dirname, _, filenames in os.walk(path):
            if len(images) >= DATASET_SIZE:
                break
            for filename in filenames:
                if len(images) >= DATASET_SIZE:
                    break
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image = Image.open(os.path.join(dirname, filename))
                    progress_bar.update(1)
                    scale = 1.4 if crop == 'cartoon' else 1.0
                    transform = tt.Compose([
                        tt.Resize(int(self.image_size * scale)),
                        tt.CenterCrop(self.image_size),
                        tt.RandomHorizontalFlip(),
                        tt.ToTensor(),
                        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                    image = transform(image)
                    images.append(image)
                    if len(images) >= DATASET_SIZE:
                        break
        return images

    def __len__(self):
        return self.size_a if self.bigger else self.size_b

    def __getitem__(self, index):
        if self.bigger:
            image_a = self.images_a[index]
            image_b = self.images_b[int(index * (self.size_b / self.size_a))]
        else:
            image_a = self.images_a[int(index * (self.size_a / self.size_b))]
            image_b = self.images_b[index]
        return image_a, image_b


# # Apples2oranges
# path_tr_a = '../input/apple2orange-dataset/trainA/'
# path_tr_b ='../input/apple2orange-dataset/trainB/'
# path_tst_a = '../input/apple2orange-dataset/testA/'
# path_tst_b = '../input/apple2orange-dataset/testB/'

# # Monet2photo
# path_tr_a = '../input/monet2photo/trainA/'
# path_tr_b ='../input/monet2photo/trainB/'
# path_tst_a = '../input/monet2photo/testA/'
# path_tst_b = '../input/monet2photo/testB/'

# Cars2toys
# path_a = '../input/stanford-cars-dataset/cars_train/cars_train/'
# path_b = '../input/toy-cars-annotated-on-yolo-format/'

# Faces2cartoons
path_a = 'bin/data_a'
path_b = 'bin/data_b'

IMAGE_SIZE = 64
BATCH_SIZE = 20
DATASET_SIZE = 125

faces_ds = ImageDS(path_a, path_b, IMAGE_SIZE)
# test_ds = ImageDS(path_tst_a, path_tst_b, IMAGE_SIZE)

train_ds, test_ds = train_test_split(faces_ds, test_size=0.2, random_state=1)

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)


def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5


def show(a_real, a_trans, a_recov,
         b_real, b_trans, b_recov,
         grid,
         rows=4):
    plt.subplot(grid[0:rows, 0:1])

    plt.imshow(denorm(make_grid(torch.cat((a_real[:rows].detach().cpu(),
                                           a_trans[:rows].detach().cpu(),
                                           a_recov[:rows].detach().cpu()),
                                          dim=0),
                                nrow=rows).permute(1, 2, 0)
                      )
               )
    plt.axis('off')
    plt.subplot(grid[0:rows, 1:2])
    plt.imshow(denorm(make_grid(torch.cat((b_real[:rows].detach().cpu(),
                                           b_trans[:rows].detach().cpu(),
                                           b_recov[:rows].detach().cpu()), dim=0), nrow=rows
                                ).permute(1, 2, 0)
                      )
               )
    plt.axis('off')

print(device)
a_real, a_trans = next(iter(train_dl))
a_recov = a_real
b_trans, b_real = next(iter(train_dl))
b_recov = b_real
print(a_real.shape, a_trans.shape, a_recov.shape)
rows = 6
plt.figure(figsize=(17, rows * 1.2))
grid = plt.GridSpec(rows + 5, 2, wspace=.1, hspace=.01)
show(a_real, a_trans, a_recov, b_real, b_trans, b_recov, grid, rows)
plt.show()