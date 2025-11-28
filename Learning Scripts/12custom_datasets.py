import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn, eval_model, print_train_time, train_step, test_step
from timeit import default_timer as timer
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torchmetrics, mlxtend
from pathlib import Path
from PIL import Image
import random
import numpy as np
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"


# random.seed(42)
image_path_list = list(image_path.glob("*/*/*.jpg"))
# print(image_path_list)
# random_image_path = random.choice (image_path_list)
# print(random_image_path)
# image_class = random_image_path.parent.stem
# print(image_class)


# img_as_array = np.asarray(img) H W C

#transforming data to tensors

data_transform = transforms.Compose([#takes pil image as input
    #resize images to 64x64
    transforms.Resize(size = (64, 64)), # less performance of model but faster processing
    transforms.RandomHorizontalFlip(p=0.5), #flip images randomly on horizontal, artifically increase image diversity(50% of the time flip)
    transforms.ToTensor() 
])

def plot_transformed_images(image_paths, transform, n=3, seed = None) :
    '''
    Selects random images from a path of images and loads/transforms them then plots the original vs transformed

    '''

    if seed:
        random.seed(seed)

    random_image_paths = random.sample(image_paths, k = n)

    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows = 1, ncols = 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis("off")

            #transformed version
            transformed_image = transform(f).permute(1,2,0) #switch from CHW to HWC
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nSize: {transformed_image.shape}")
            ax[1].axis("off")


            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize = 16)
            fig.show()

plot_transformed_images(
    image_paths=image_path_list,
    transform=data_transform,
    n=3,
    seed = 42
)



input = input("pause")

#option 1 loading image with image folder class
train_data = datasets.ImageFolder(
    root=train_dir,
    transform = data_transform, #data transform
    target_transform = None #label transform
)

test_data = datasets.ImageFolder(
    root=test_dir,
    transform=data_transform
)


# print(train_data, test_data)

class_names = train_data.classes
class_dict = train_data.class_to_idx


print(len(train_data), len(test_data))

# print(os.cpu_count())
#num_workers is # of cpu core
train_dataloader = DataLoader(dataset=train_data, batch_size = BATCH_SIZE, num_workers=1, shuffle = True)
test_dataloader = DataLoader(dataset=test_data, batch_size = BATCH_SIZE, num_workers=1, shuffle = False)




#option 2 loading image data with a custom data set loader






