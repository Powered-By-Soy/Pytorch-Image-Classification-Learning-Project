import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
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
from typing import Tuple, Dict, List

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

target_directory = train_dir

class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])

print(class_names_found)

def find_classes(directory: str) ->Tuple[List[str], Dict[str, int]]:
    #finds class fpr;der names in target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}")

    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    return classes, class_to_idx

print(find_classes(target_directory))


class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir: str, transform=None):
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: int) -> Image.Image:
        "opens image and returns"
        image_path = self.paths[index]
        return Image.open(image_path)
    
    #override len and getitem (required)
    def __len__(self) -> int:
        "returns total number of samples"
        return len(self.paths)

    def __getitem__(self, index: int)-> Tuple[torch.Tensor, int]:
        "returns one sample of data, data and label (X,y)"
        img = self.load_image(index)
        class_name = self.paths[index].parent.name #expects path in format: data_folder/class_name/image.jpg