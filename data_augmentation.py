

import os
import cv2
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import random
randZoom = random.randint(350, 450)



input_path = 'C:\\Users\\gui_m\\.spyder-py3\\circDeep-master\\data\\brain-tumour-segmentation-and-classification\\data\\Inputs\\'
mask_path = 'C:\\Users\\gui_m\\.spyder-py3\\circDeep-master\\data\\brain-tumour-segmentation-and-classification\\data\\Masks\\'

slices = ['Axiale','Coronale','Sagittale']

input_output_path = 'C:\\Users\\gui_m\\.spyder-py3\\circDeep-master\\data\\brain-tumour-segmentation-and-classification\\augmented_data\\Inputs\\'
mask_output_path = 'C:\\Users\\gui_m\\.spyder-py3\\circDeep-master\\data\\brain-tumour-segmentation-and-classification\\augmented_data\\Masks\\'



transform = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Transpose(),
        A.ShiftScaleRotate(),
        A.CropNonEmptyMaskIfExists(randZoom,randZoom),
        A.GridDropout(ratio=0.33,holes_number_x=5,holes_number_y=5,fill_value=[84,1,68],mask_fill_value=0),
        A.ShiftScaleRotate(),
        A.Resize(512,512, always_apply=True)
                ## Check transpose_mask=True
    ])



def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)




def get_train_augmentation(size=512):
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.ShiftScaleRotate(),
        A.CoarseDropout(max_holes=8, max_height=10, max_width=10, mask_fill_value=0, always_apply=True), # For visualization, set always_apply=True.
        A.Resize(size,size, always_apply=True),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ## Check transpose_mask=True
        ToTensorV2(transpose_mask=True)
    ])

def get_valid_augmentation(size=512):
    return A.Compose([
        A.Resize(size,size, always_apply=True),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ## Check transpose_mask=True
        ToTensorV2(transpose_mask=True)
    ])

# img = cv2.imread('C:\\Users\\gui_m\\.spyder-py3\\circDeep-master\\data\\brain-tumour-segmentation-and-classification\\data\\test_augmentation\\2.png')
# mask = cv2.imread('C:\\Users\\gui_m\\.spyder-py3\\circDeep-master\\data\\brain-tumour-segmentation-and-classification\\data\\test_augmentation\\2_mask.png')


# augmented = transform(image=img, mask=mask)


# image_center_cropped = augmented['image']
# mask_center_cropped = augmented['mask']
# image_padded = augmented['image']
# mask_padded = augmented['mask']
# print(image_center_cropped.shape, mask_center_cropped.shape)


# print(len(augmented))
#visualize(image_padded, mask_padded, original_image=img, original_mask=mask)
print(len(os.listdir(input_path+'Axiale')))

def augment_slice(orientation='Axiale'):
    counter = 29364
    for filename in os.listdir(input_path+orientation):
        if filename.endswith("png"):
            img = cv2.imread(input_path+orientation+'\\' + filename)    
            mask = cv2.imread(mask_path+filename)
            for i in range(20):
                randZoom = 500
                transform = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Transpose(),
        A.ShiftScaleRotate(),
        A.CropNonEmptyMaskIfExists(randZoom,randZoom),
        A.ShiftScaleRotate(),
        A.Resize(512,512, always_apply=True)
                ## Check transpose_mask=True
    ])
                augmented = transform(image=img, mask=mask)
                image_center = augmented['image']
                mask_center = augmented['mask']
                cv2.imwrite('%i' % counter + '.png', image_center)
                cv2.imwrite('mask_' + '%i' % counter + '.png', mask_center)
                counter += 1
            counter +=1
                
augment_slice()            


# fig, ax = plt.subplots(figsize=(16, 8),  nrows=1, ncols=2)
# ax[0].imshow(image_center_cropped)
# ax[1].imshow(mask_center_cropped)












