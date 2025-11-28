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

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.FashionMNIST(
    root = "data", #where to download data to
    train = True, #do we want test or training data
    download = True, #do we want to download
    transform=torchvision.transforms.ToTensor(),#how do we want to transform data
    target_transform=None #how do we want to transform the labels
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform=ToTensor(),
    target_transform=None
)
class_names = train_data.classes
train_dataloader = DataLoader(dataset=train_data, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(dataset=test_data, batch_size = BATCH_SIZE, shuffle = False)
class FashionMNISTModelV2(nn.Module):
    '''
    model from CNN explainer website

    kernel is dimensions of squares convolving around image

    stride is number of squares the kernel moves

    padding is the amount of extra squares added to the edges of the image

    each layer compresses the image
    '''
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential( #2d for 2d data, 3d for 3d datas, 1d for 1d data
            nn.Conv2d(in_channels = input_shape, out_channels = hidden_units, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #takes the max value of a 2x2 square, kernal = 2 means tuple (2,2) or can pass tuple
            #due to kernel size of 2x2, final dimension would result in half the width and height after processing
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(hidden_units*49),out_features=output_shape)
        )
        #in features can be calculated
        #best practice is to print shape of output before classifier and figure out dimensions from Flatten behavior

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        return x


torch.manual_seed(42)
#based on tiny VGG, tiny VGG is coloured so input shape = 3, we are using gray scale so input shape = 1

loaded_model_2 = FashionMNISTModelV2(input_shape = 1, hidden_units = 10, output_shape = len(class_names)).to(device)
# print(loaded_model_2.state_dict())

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True, exist_ok = True)

MODEL_NAME = "pytorch_computer_vision_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
# print("loaded")
# print(loaded_model_2.state_dict())


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = loaded_model_2.parameters(), lr = 0.1)


model_2_results = eval_model(model=loaded_model_2, 
                            data_loader= test_dataloader,
                            loss_fn = loss_fn,
                            accuracy_fn = accuracy_fn,
                            device = device)
print(model_2_results) 

