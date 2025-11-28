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
class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                input_shape: int,
                hidden_units: int,
                output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape,
                      out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units,
                      out_features = output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

torch.manual_seed(42)

model_1 = FashionMNISTModelV1(input_shape = 784,
                                hidden_units = 10,
                                output_shape = len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_1.parameters(), lr = 0.1)


torch.manual_seed(42)

train_time_start_on_gpu = timer()
epochs = 3

for epoch in tqdm(range(epochs)):

    print(f"Epoch: {epoch}\n")
    train_step(
        model = model_1,
        data_loader = train_dataloader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        accuracy_fn = accuracy_fn,
        device = device
    )

    test_step(
        model = model_1,
        data_loader = train_dataloader,
        loss_fn = loss_fn,
        accuracy_fn = accuracy_fn,
        device = device
    )

    train_time_end_on_gpu = timer()

    total_train_time_model_1 = print_train_time(start = train_time_start_on_gpu, end = train_time_end_on_gpu, device = device)

#sometimes training on gpu is slower than cpu due to overhead of moving to gpu

model_1_results = eval_model(model=model_1, 
                            data_loader= test_dataloader,
                            loss_fn = loss_fn,
                            accuracy_fn = accuracy_fn,
                            device = device)
print(model_1_results)



'''
epoch time
Train time on cuda: 14.527 seconds

Train time on cuda: 30.423 seconds
Train time on cuda: 45.676 seconds

{'model_name': 'FashionMNISTModelV1', 'model_loss': 0.6874349117279053, 'model_acc': 75.65894568690096}
'''

